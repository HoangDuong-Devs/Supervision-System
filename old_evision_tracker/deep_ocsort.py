import os
from typing import List

import numpy as np

from configs.autocfg import cfg
from modules.logging.logger import Logger
from modules.qdrant.qdrant_reid import Qdrant, QdrantReId
from tracker.appearance.reid_triton_server import ReIDDetectMultiBackend
from tracker.motion.cmc import get_cmc_method
from tracker.trackers.deepocsort.assign_manager import AssignTree
from tracker.trackers.deepocsort.kalman_tracker import KalmanBoxTracker
from tracker.trackers.deepocsort.track_interaction import TrackInteractor
from tracker.trackers.deepocsort.utils import *
from tracker.utils.association import associate, linear_assignment
from tracker.utils.iou import get_asso_func


def find_key(dictionary, value):
    key_list = list(dictionary.keys())
    val_list = list(dictionary.values())
    position = val_list.index(value)
    return key_list[position]


class DeepOCSort:
    def __init__(
        self,
        device,
        per_class=True,
        det_thresh=cfg.REID.DET_THRESHOLD,
        max_age=cfg.REID.MAX_AGE,
        min_hits=cfg.REID.MIN_HITS,
        iou_threshold=cfg.REID.IOU_THRESHOLD,
        delta_t=cfg.REID.DELTA_T,
        asso_func="iou",
        inertia=cfg.REID.INERTIA,
        w_association_emb=0.5,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        new_kf_off=False,
        cameraID=None,
        **kwargs,
    ):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = get_asso_func(asso_func)
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        self.per_class = per_class
        # KalmanBoxTracker.count = 1
        self.logger = Logger()
        self.model = ReIDDetectMultiBackend(device=device)
        # "similarity transforms using feature point extraction, optical flow, and RANSAC"
        self.cmc = get_cmc_method("sof")()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        self.new_kf_off = new_kf_off
        self.message_det = dict()
        self.upsert_data = []
        self.ids = []

        # system attributes
        self.cameraID = cameraID
        self.qdrant_client = Qdrant(
            host=cfg.REID_QDRANT.HOST, port=cfg.REID_QDRANT.PORT
        )
        self.qdrant_reid = QdrantReId(cfg.REID_QDRANT.HOST, port=cfg.REID_QDRANT.PORT)
        # if kwargs['running_mode'] == 'test':
        #     self.collection_name = os.environ['COLLECTION_NAME']
        # else:
        self.collection_name = cfg.COMPANY.NAME

        # connect-update
        self.qdrant_client.create(collection_name=self.collection_name)
        self.id_range = kwargs.get("id_range", [0, 1000])
        KalmanBoxTracker.count = self.id_range[0]
        print(
            f"DeepOCSORT parameters: collection_name: {self.collection_name}| id_range: {self.id_range}"
        )
        self.assigntree = AssignTree()
        self.struct = payload_struct
        self.coexistence_ids = dict()
        self.previous_unmatched_trks = []
        self.removed_keys = []
        self.tuple_suspect_swapped_idx = []

        self.trackinteractor = TrackInteractor(
            qdrant_client=self.qdrant_client,
            colection_name=self.collection_name,
            id_range=self.id_range,
            logger=self.logger,
            cameraID=cameraID,
            max_age=max_age,
        )
        self.max_cr_count = self.id_range[0]
        self.count = self.max_cr_count + 1
        self.init_attr_trigger = True

    def __init_attr__(self):
        self.max_cr_count = self.trackinteractor.get_max_cr_count(self.cameraID)
        self.count = self.max_cr_count + 1
        self.init_attr_trigger = False

    def update(self, dets, img, **kwargs):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.frame_count += 1
        self.height, self.width = img.shape[:2]

        scores = dets[:, 4]
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        assert dets.shape[1] == 7
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # appearance descriptor extraction
        if self.embedding_off or dets.shape[0] == 0:
            dets_embs = np.ones((dets.shape[0], 1))
        else:
            # (Ndets x X) [512, 1024, 2048]
            dets_embs = self.model.get_features(dets[:, 0:4], img)

        # CMC
        if not self.cmc_off:
            transform = self.cmc.apply(img, dets[:, :4])
            for trk in self.trackers:
                trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        dets_alpha = af + (1 - af) * (1 - trust)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        if len(trk_embs) > 0:
            trk_embs = np.vstack(trk_embs)
        else:
            trk_embs = np.array(trk_embs)

        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.trackers
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.trackers
            ]
        )

        """
            First round of association
        """
        # (M detections X N tracks, final score)
        if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
            stage1_emb_cost = None
        else:
            stage1_emb_cost = dets_embs @ trk_embs.T
        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5],
            trks,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            stage1_emb_cost,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
        )
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        """
            Second round of associaton by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]

            iou_left = self.asso_func(left_dets, left_trks)
            # TODO: is better without this
            emb_cost_left = left_dets_embs @ left_trks_embs.T
            if self.embedding_off:
                emb_cost_left = np.zeros_like(emb_cost_left)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    try:
                        iou_left = iou_left[m[0], m[1]]
                    except:
                        return np.array([])
                    if iou_left.ndim == 2:
                        iou_left = 0.7
                    if (iou_left < self.iou_threshold).any():
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    self.trackers[trk_ind].update_emb(
                        dets_embs[det_ind], alpha=dets_alpha[det_ind]
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, np.array(to_remove_det_indices)
                )
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i],
                delta_t=self.delta_t,
                emb=dets_embs[i],
                alpha=dets_alpha[i],
                new_kf=not self.new_kf_off,
                cr_count=self.count,
            )
            self.trackers.append(trk)
            self.count += 1
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(
                    np.concatenate(
                        (d, [trk.id], [trk.conf], [trk.cls], [trk.det_ind])
                    ).reshape(1, -1)
                )
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])

    def update_reid(self, dets, img, **kwargs):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input type '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input type '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        if self.init_attr_trigger:
            self.__init_attr__()

        frame_idx = kwargs.get("frame_idx")
        self.logger.info(info="Info v3", key=f"frame {frame_idx}")

        self.frame_count += 1
        self.height, self.width = img.shape[:2]

        scores = dets[:, 4]
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        assert dets.shape[1] == 7
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # appearance descriptor extraction
        if self.embedding_off or dets.shape[0] == 0:
            self.dets_embs = np.ones((dets.shape[0], 1))
        else:
            # (Ndets x X) [512, 1024, 2048]
            self.dets_embs = self.model.get_features(dets[:, 0:4], img)

        # CMC
        if not self.cmc_off:
            transform = self.cmc.apply(img, dets[:, :4])
            for trk in self.trackers:
                trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        dets_alpha = af + (1 - af) * (1 - trust)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        if len(trk_embs) > 0:
            trk_embs = np.vstack(trk_embs)
        else:
            trk_embs = np.array(trk_embs)

        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.trackers
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.trackers
            ]
        )

        """
            First round of association
        """
        # (M detections X N tracks, final score)
        if self.embedding_off or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
            stage1_emb_cost = None
        else:
            stage1_emb_cost = self.dets_embs @ trk_embs.T
        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5],
            trks,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            stage1_emb_cost,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
        )
        for m in matched:
            det_idx = m[0]
            trk_idx = m[1]
            self.trackers[trk_idx].update(dets[det_idx, :])
            self.trackers[trk_idx].update_emb(
                self.dets_embs[det_idx], alpha=dets_alpha[det_idx]
            )

        self.trackinteractor.update_attributes(
            frame_count=frame_idx,
            ids=self.ids,
            trackers=self.trackers,
            coexistence_ids=self.coexistence_ids,
            tuple_suspect_swapped_idx=self.tuple_suspect_swapped_idx,
        )

        self.trackers = self.trackinteractor.process_matched_info(
            trackers=self.trackers, matched=matched, dets=dets, dets_embs=self.dets_embs
        )

        """
            Second round of associaton by OCR #TODO: check LongVD's reid logic (deepocsort_v1) of unmatched process
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = self.dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]

            iou_left = self.asso_func(left_dets, left_trks)
            # TODO: is better without this
            emb_cost_left = left_dets_embs @ left_trks_embs.T
            if self.embedding_off:
                emb_cost_left = np.zeros_like(emb_cost_left)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    try:
                        iou_left = iou_left[m[0], m[1]]
                    except:
                        return np.array([])
                    if iou_left.ndim == 2:
                        iou_left = 0.7
                    if iou_left < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    self.trackers[trk_ind].update_emb(
                        self.dets_embs[det_ind], alpha=dets_alpha[det_ind]
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, np.array(to_remove_det_indices)
                )
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        upsert_data = []
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i],
                delta_t=self.delta_t,
                emb=self.dets_embs[i],
                alpha=dets_alpha[i],
                new_kf=not self.new_kf_off,
                cr_count=self.count,
            )
            self.count += 1
            self.trackers.append(trk)
            data, uuid = self.trackinteractor.create_upsert_data(
                self.dets_embs[i], trk.id, dets[i]
            )
            upsert_data.append(data)

        if len(upsert_data):
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                list_data=upsert_data,
                payload_struct=self.struct,
            )
        i = len(self.trackers)
        self.cr_ids = []

        self.logger.add(
            f"unmatched_trks {unmatched_trks} previous_unmatched_trks {self.previous_unmatched_trks}"
        )
        new_hidden_trk_idxs = list(
            set(unmatched_trks) - set(self.previous_unmatched_trks)
        )

        suspect_swapped_trks = []
        if len(new_hidden_trk_idxs):
            new_hidden_bboxes = [
                self.trackers[i].last_observation[0:4] for i in new_hidden_trk_idxs
            ]
            cr_bboxes = [self.trackers[m[1]].last_observation[0:4] for m in matched]
            matched_trk_idxs = [m[1] for m in matched]
            self.logger.add(f"hidden-cr {new_hidden_trk_idxs} {matched_trk_idxs}")
            if len(new_hidden_bboxes) and len(cr_bboxes):
                ipbs = intersection_percent_batch(
                    new_hidden_bboxes, cr_bboxes, symmetry=False
                )
                self.logger.add(f"ipbs {new_hidden_bboxes} {cr_bboxes}")
                for idx, ipb in enumerate(ipbs):
                    if ipb > cfg.REID.THRESHOLD.SET_SUSPECT_INTERSECTION_OF_HIDDEN_BBOX:
                        for idx_hidden in new_hidden_trk_idxs:
                            if (
                                idx_hidden,
                                matched[idx][1],
                            ) not in self.tuple_suspect_swapped_idx or (
                                matched[idx][1],
                                idx_hidden,
                            ) not in self.tuple_suspect_swapped_idx:
                                self.tuple_suspect_swapped_idx.append(
                                    (idx_hidden, matched[idx][1])
                                )
                                suspect_swapped_trks.append(
                                    (idx_hidden, matched[idx][1])
                                )

                for tuple in suspect_swapped_trks:
                    if self.trackers[tuple[0]].suspect is False:
                        self.trackers[tuple[0]].set_suspect()
                    else:
                        self.trackers[tuple[0]].reset_suspect(True)
                    if self.trackers[tuple[1]].suspect is False:
                        self.trackers[tuple[1]].set_suspect()
                    else:
                        self.trackers[tuple[1]].reset_suspect(True)

        self.logger.add(f"suspect swapped {self.tuple_suspect_swapped_idx}")
        self.previous_unmatched_trks = unmatched_trks

        removed_trigger = False
        to_del_keys = []
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(
                    np.concatenate(
                        (
                            d,
                            [trk.id],
                            [trk.conf],
                            [trk.cls],
                            [trk.det_ind],
                            [trk.suspect],
                        )
                    ).reshape(1, -1)
                )
                self.cr_ids.append(trk.id)
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                to_del_keys.append(i)
                self.logger.add(f"remove tracker idx {i}")
                removed_trigger = True

        # update after remove tracker
        if removed_trigger is True:
            (
                self.tuple_suspect_swapped_idx,
                _,
                _,
            ) = self.trackinteractor.remove_dead_attributes(to_del_keys)

        ################################################################
        self.ids = [trk.id for trk in self.trackers]
        for id in self.cr_ids:
            cr_ids = self.cr_ids.copy()
            if id not in self.coexistence_ids.keys():
                self.coexistence_ids[id] = []
            else:
                cr_ids.remove(id)
                self.coexistence_ids[id] = self.coexistence_ids[id] + cr_ids
                unique_list = list(set(self.coexistence_ids[id]))
                unique_list.sort()
                self.coexistence_ids[id] = unique_list

        self.logger.add(f"coexistance {self.coexistence_ids}")
        self.logger.add("end-------------------------------")
        self.logger.save(f"logs/logfile_reid_{self.cameraID}.txt")
        ################################################################
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])

    def set_suppection(self):
        pass