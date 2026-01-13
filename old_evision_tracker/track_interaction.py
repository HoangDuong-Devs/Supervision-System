
"""
_____________________________________________________________________________
Created By  : Thinh Nguyen-Quang
Created Date: 21/12/2023 VNT
_____________________________________________________________________________
"""

import json
from datetime import datetime
from typing import List

import numpy as np
import pika
from dotenv import find_dotenv, load_dotenv

from configs.autocfg import cfg
from modules.qdrant.qdrant_reid import Qdrant, QdrantReId, rematching
from tracker.trackers.deepocsort.assign_manager import AssignIDManager
from tracker.trackers.deepocsort.kalman_tracker import KalmanBoxTracker
from tracker.trackers.deepocsort.utils import *
from tracker.trackers.deepocsort.vector_manager import VectorManager
from tracker.utils import logger as LOGGER
from tracker.utils.association import linear_assignment
from tracker.utils.iou import iou_batch

class RuleKeeper(object):
    def __init__(self):
        self.new_data_status = 1
        self.old_data_status = 0

rule = RuleKeeper()

class QdrantInteractor(object):
    def __init__(self):
        self.camera_id = None
        self.collection_name = None
        self.cr_count = None

    def pull_data(self, **kwargs):
        data_status = kwargs.get("new_data", rule.old_data_status)
        if self.camera_id is not None:
            qdr_trks, _ = self.qdrant_client.get_data(
                self.collection_name, status=data_status, camera_id=self.camera_id
            )
        else:
            qdr_trks, _ = self.qdrant_client.get_data(
                self.collection_name, status=data_status
            )

        # get bbox_ids/counts
        qdr_trk_uuids = [qdr_trks[i].id for i in range(len(qdr_trks))]
        qdr_bbox_ids = [qdr_trks[i].payload["bbox_id"] for i in range(len(qdr_trks))]
        qdr_cr_count = [qdr_trks[i].payload["cr_count"] for i in range(len(qdr_trks))]

        self.qdr_trk_uuids = self.qdr_trk_uuids + qdr_trk_uuids
        self.qdr_bbox_ids = self.qdr_bbox_ids + qdr_bbox_ids

    def __call__(self):
        pass


class TrackMessageProducer(object):
    """_summary_

    Args:
        object (_type_): _description_
    ****************
    The main idea belongs to LongVD
    """

    def __init__(self, host=None, port=8701) -> None:
        self.__host, self.__port = host, port
        self.connect(self.__host, self.__port)
        self.set_channel()

    def connect(self, host, port):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host,
                port,
                credentials=pika.PlainCredentials(
                    os.environ["USERRABBIT_DOCKER"], os.environ["PWDSRABBIT_DOCKER"]
                ),
                heartbeat=10,
            )
        )

    def set_channel(self):
        self.__channel = self.connection.channel()

        self.__exchange_name = os.environ["EXCHANGE"]
        exchange_type = os.environ["EXCHANGE_TYPE"]
        queue = os.environ["QUEUE"]
        self.__routing_key = os.environ["ROUTING_KEY"]

        try:
            self.__channel.exchange_declare(
                exchange=self.__exchange_name,
                exchange_type=exchange_type,
            )
            self.__channel.queue_declare(queue=queue)
        except Exception as e:
            self.__channel = self.connection.channel()

    def process_images(self, image_str: str):
        return image_str

    def create_track_messs(self, max_cr_count, cameraID, message_det):
        message = {
            "timestamp": datetime.now().isoformat(),
            "current_count": max_cr_count,
            "camera_id": cameraID,
            "trackers": message_det,
        }

        return message

    def send(self, input: dict):
        input = json.dumps(input)

        if not self.connection or self.connection.is_closed:
            self.connect(self.__host, self.__port)
            self.set_channel()
        self.__channel.basic_publish(
            exchange=self.__exchange_name,
            routing_key=self.__routing_key,
            body=input,
        )


def find_key(dictionary, value):
    key_list = list(dictionary.keys())
    val_list = list(dictionary.values())
    position = val_list.index(value)
    return key_list[position]

class TrackInteractor():
    def __init__(
        self,
        qdrant_client: Qdrant = Qdrant(host=cfg.REID_QDRANT.HOST, port=cfg.REID_QDRANT.PORT),
        colection_name: str = None,
        **kwargs
        ) -> None:
        LOGGER.info(f"TrackInterator is initialized !")
        print("TrackInterator is initialized !")
        
        # init parameters
        self.cameraID = kwargs.get('cameraID', None)
        self.logger = kwargs.get('logger', None)
        self.id_range = kwargs.get('id_range')
        self.max_age = kwargs.get('max_age')
        
        self.host = cfg.REID_QDRANT.HOST
        self.port = cfg.REID_QDRANT.PORT
        self.qdrant_client = qdrant_client
        self.collection_name = colection_name
        self.struct = payload_struct

        print(f"TrackInterator parameters: host: {self.host}| port: {self.port}| collection_name: {self.collection_name}| id_range: {self.id_range}")
        # init attributed parameter
        self.max_bbox_id = 0
        self.init = True
        self.init_data_phase = True
        self.previous_matched = []
        self.frame_count = 0
        self.ids = []
        self.coexistence_ids = dict()
        
        self.max_cr_count = self.get_max_cr_count()
        self.exist_id_range = (self.id_range[0], self.max_cr_count)
                
        # self.assign_manager.local_assignment = dict()
        # self.assign_manager.approved_assignment = dict()
        self.tuple_suspect_swapped_idx = []
        self.vector_uuids = dict()
        self.counters = dict()
        self.to_del = []
        
        self.vector_manager = VectorManager(
            qdrant_client=self.qdrant_client,
            collection_name=self.collection_name,
            exist_id_range=self.exist_id_range,
            logger = self.logger
            )
        
        self.assign_manager = AssignIDManager(
            qdrant_client= self.qdrant_client,
            collection_name=self.collection_name,
            logger=self.logger,
            max_age = self.max_age
            )
        self.vector_manager.sync()
        
    def update_attributes(self, **kwargs):
        self.frame_count = kwargs.get('frame_count', self.frame_count)
        self.ids = kwargs.get('ids', self.ids)
        self.coexistence_ids = kwargs.get('coexistence_ids', self.coexistence_ids)
        # self.assign_manager.local_assignment = kwargs.get('local_assignment')
        # self.assign_manager.approved_assignment = kwargs.get('approved_assignment')
        self.tuple_suspect_swapped_idx = kwargs.get('tuple_suspect_swapped_idx', self.tuple_suspect_swapped_idx)
        
        #update attributes of attribute class
        self.assign_manager.update_attributes(
            tuple_suspect_swapped_idx=self.tuple_suspect_swapped_idx,
            ids=self.ids,
            coexistence_ids=self.coexistence_ids
            )
        
 
    def get_max_cr_count(self, camera_id: str = None):
        if camera_id is not None:
            data, not_empty = self.qdrant_client.get_data(
                collection_name=self.collection_name,
                type="storage",
                camera_id=self.cameraID,
            )
        else:
            data, not_empty = self.qdrant_client.get_data(
                collection_name=self.collection_name,
                type="storage",
            )

        if not_empty:
            cr_counts = [data.payload["bbox_id"] for data in data]
            max_cr_count = max(cr_counts)
        else:
            max_cr_count = self.id_range[0]

        return max_cr_count

    def create_upsert_data(self, det_emb, bbox_id, det):
        uuid =  self.qdrant_client.gen_uuid()
        data = {
            "vector": det_emb,
            "id": uuid,
            "bbox_id": bbox_id,
            "camera_id": self.cameraID,
            "dets": det.tolist(),
            "cr_count": self.max_cr_count,
            # "first_registration": False,
            "type": "storage"
        }
        self.vector_manager.add_to_manager(uuid=uuid, key=bbox_id)
        return data, uuid
    
    def process_matched_info(self, trackers, matched, dets, dets_embs):
        intersection_percent = intersection_percent_batch(dets[:, :4], dets[:, :4])
        self.assign_manager.local_assignment = dict(sorted(self.assign_manager.local_assignment.items()))
        self.assign_manager.approved_assignment = dict(sorted(self.assign_manager.approved_assignment.items()))
        infer_major = False
        approvement_trigger = False

        # set suspect
        for m in matched:
            det_idx = m[0]; trk_idx = m[1]
            if (
                intersection_percent[det_idx] > cfg.REID.THRESHOLD.SET_SUSPECT_INTERSECTION 
                and trackers[trk_idx].suspect is False
                ):
                trackers[trk_idx].reset_suspect(True)
                self.logger.add(f"reset suspect to True")

        # search suspect object
        self.assign_manager.assign_idx_request = []
        for m in matched:
            det_idx = m[0]; trk_idx = m[1]
            if trk_idx not in self.assign_manager.local_assignment.keys():
                self.assign_manager.approved_assignment[trk_idx] = False
                self.assign_manager.local_assignment[trk_idx] = (trackers[trk_idx].id, None, 0)
            if trackers[trk_idx].suspect is True:
                if intersection_percent[det_idx] < cfg.REID.THRESHOLD.SUSPECT_SEARCH_ACCECPT:
                    trackers[trk_idx].update_suspect(infer_major=infer_major)
                    old_id_candidates, rematched, searched_results, not_empty = (
                        self.rematch_id(
                            det_emb=dets_embs[det_idx],
                            threshold=cfg.REID.THRESHOLD.REMATCH_WITH_LOW_INTERSECTION ,
                            # ignore_id=trackers[trk_idx].id,
                            best_point=False,
                            check_exists=False,
                        )
                    )
                    if old_id_candidates is not None:
                        trackers[trk_idx].suspect_data["suspect_id"] = (
                            trackers[trk_idx].suspect_data["suspect_id"]
                            + old_id_candidates
                        )
                else:
                    old_id_candidates, rematched, searched_results, not_empty = (
                        self.rematch_id(
                            det_emb=dets_embs[det_idx],
                            threshold=cfg.REID.THRESHOLD.REMATCH_WITH_HIGH_INTERSECTION ,
                            # ignore_id=trackers[trk_idx].id,
                            best_point=False,
                            check_exists=False,
                        )
                    )
                    if old_id_candidates is not None:
                        trackers[trk_idx].suspect_data["suspect_id"] = (
                            trackers[trk_idx].suspect_data["suspect_id"]
                            + old_id_candidates
                        )
            else:  # trackers[trk_idx].suspect is False
                if infer_major: major, count = trackers[trk_idx].suspect_data["suspect_id"]
                else:           major, count = majority_element(trackers[trk_idx].suspect_data["suspect_id"])
                approvement_trigger = True
                self.assign_manager.local_assignment.update({trk_idx: (trackers[trk_idx].id, major, count)})
                self.assign_manager.assign_idx_request.append(trk_idx)
        ################################################################
        
        # Process cases and update approved_assignments based on expected_assignments
        if approvement_trigger:
            
            trackers = self.assign_manager.approve_suspect_case_1(trackers)
            trackers = self.assign_manager.approve_suspect_case_2(trackers)
            trackers = self.assign_manager.approve_suspect_case_3(trackers)
            self.logger.add(f"self.assign_manager.local_assignment {self.assign_manager.local_assignment}")
            self.logger.add(f"self.assign_manager.approved_assignment {self.assign_manager.approved_assignment}")
            trackers = self.assign_manager.assign_id(trackers)
            trackers = self.assign_manager.update_trk_attributes(trackers)
            self.logger.add(f"self.assign_manager.local_assignment {self.assign_manager.local_assignment}")
            self.logger.add(f"self.assign_manager.approved_assignment {self.assign_manager.approved_assignment}")
            self.logger.add(f"assigntree {self.assign_manager.assigntree.assigntree}")
            approvement_trigger = False
        trackers = self.assign_manager.assign_to_root(trackers)
        
        # push matched data to qdrant
        upsert_data = []
        for m in matched:
            det_idx = m[0]; trk_idx = m[1]
            if trackers[trk_idx].suspect is False and self.frame_count % 1 == 0:
                if intersection_percent[det_idx] < cfg.REID.THRESHOLD.UPLOAD:
                    det_emb=dets_embs[det_idx]; bbox_id=trackers[trk_idx].id; det=dets[det_idx]
                    data, uuid =  self.create_upsert_data(det_emb=det_emb, bbox_id=bbox_id, det=det)
                    upsert_data.append(data)
                    self.vector_manager.add_to_manager(uuid=uuid, key=bbox_id)
                self.vector_manager.manage_vector()
            else: pass
                    
        if len(upsert_data):
            self.qdrant_client.upsert(
                collection_name=self.collection_name, list_data=upsert_data, payload_struct=self.struct
            )
        
        return trackers
        ################################################################

    def rematch_id(
        self,
        det_emb: np.ndarray,
        exist_ids: List = [],
        threshold: float = cfg.REID.THRESHOLD.REMATCH_WITH_HIGH_INTERSECTION ,
        search_global: bool = False,
        best_point: bool = True,
        **kwargs,
    ):
        ignore_id = kwargs.get("ignore_id", False)
        check_exists = kwargs.get("check_exists", False)
        matched_bbox_ids = kwargs.get("matched_bbox_ids", [])

        if ignore_id:
            must_not = {"bbox_id": ignore_id}
        else:
            must_not = {"bbox_id": 0}
        if search_global:
            searched_results, not_empty = self.qdrant_client.search_v1(
                collection_name=self.collection_name,
                vector=det_emb,
                threshold=threshold,
                limit=5,
                type="storage",
                must_not=must_not,
            )
        else:
            searched_results, not_empty = self.qdrant_client.search_v1(
                collection_name=self.collection_name,
                vector=det_emb,
                threshold=threshold,
                limit=5,
                type="storage",
                camera_id=self.cameraID,
                must_not=must_not,
            )
        if not_empty:
            if best_point:
                old_bbox_id = searched_results[0].payload["bbox_id"]
                if old_bbox_id in exist_ids and check_exists == True:
                    print(f"failed rematch by existing bbox_id: {old_bbox_id}")
                    case = 1
                    rematched = False
                else:
                    case = 2
                    rematched = True
            else:
                old_bbox_id = [
                    searched_result.payload["bbox_id"]
                    for searched_result in searched_results
                ]
                rematched = True
        else:
            case = 3
            old_bbox_id = None
            rematched = False

        return old_bbox_id, rematched, searched_results, not_empty

    def remove_dead_attributes(self, keys):
        for key in keys:
            del self.assign_manager.local_assignment[key]
            del self.assign_manager.approved_assignment[key]
        local_assignment = dict()
        approved_assignment = dict()
        update_swapped_idxs = dict()
        for idx, (key, value) in enumerate(self.assign_manager.local_assignment.items()):
            local_assignment.update({idx: value})
            approved_assignment.update({idx: False})
            update_swapped_idxs.update({key: idx})
        self.assign_manager.local_assignment = local_assignment
        self.assign_manager.approved_assignment = approved_assignment
        
        temp = []
        for idx, tuple in enumerate(self.tuple_suspect_swapped_idx):
            if tuple[0] in update_swapped_idxs.keys() and tuple[1] in update_swapped_idxs.keys():
                temp.append((update_swapped_idxs[tuple[0]], update_swapped_idxs[tuple[1]]))
            else:
                pass
        self.tuple_suspect_swapped_idx = temp
        self.logger.add(f"suspect swapped update {self.tuple_suspect_swapped_idx}")
        return self.tuple_suspect_swapped_idx, self.assign_manager.local_assignment, self.assign_manager.approved_assignment
            