import numpy as np

from configs.autocfg import cfg
from tracker.motion.cmc import get_cmc_method
from tracker.motion.kalman_filters.adapters import OCSortKalmanFilterAdapter
from tracker.trackers.deepocsort.utils import *
from tracker.utils.association import associate, linear_assignment
from tracker.utils.iou import get_asso_func


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, det, delta_t=3, emb=None, alpha=0, new_kf=False, cr_count: int = None):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model

        self.new_kf = new_kf
        bbox = det[0:5]
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]

        if cr_count is not None:
            KalmanBoxTracker.count = cr_count
        if new_kf:
            self.kf = OCSortKalmanFilterAdapter(dim_x=8, dim_z=4)
            self.kf.F = np.array(
                [
                    # x y w h x' y' w' h'
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                ]
            )
            _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
            self.kf.P = new_kf_process_noise(w, h)
            self.kf.P[:4, :4] *= 4
            self.kf.P[4:, 4:] *= 100
            # Process and measurement uncertainty happen in functions
            self.bbox_to_z_func = convert_bbox_to_z_new
            self.x_to_bbox_func = convert_x_to_bbox_new
        else:
            self.kf = OCSortKalmanFilterAdapter(dim_x=7, dim_z=4)
            self.kf.F = np.array(
                [
                    # x  y  s  r  x' y' s'
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ]
            )
            self.kf.R[2:, 2:] *= 10.0
            self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
            self.kf.P *= 10.0
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            self.bbox_to_z_func = convert_bbox_to_z
            self.x_to_bbox_func = convert_x_to_bbox

        self.kf.x[:4] = self.bbox_to_z_func(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        # KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        # Used for OCR
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        # Used to output track after min_hits reached
        self.history_observations = []
        # Used for velocity
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t

        self.emb = emb

        self.frozen = False
        self.set_suspect()

    def set_suspect(self, status: bool = True):
        self.suspect = status
        self.suspect_data = {"suspect_id": [], "suspect_time": 0, "major": (None, 0)}
        
    def reset_suspect(self, status: bool = False):
        self.set_suspect(status)
        # self.suspect_data = {"suspect_id": [], "suspect_time": 0, "major": (None, 0)}
    
    def update_suspect(self, infer_major: bool = False):
        self.suspect_data["suspect_time"] = self.suspect_data["suspect_time"] + 1
        if infer_major:
            self.suspect_data["major"] = majority_element(self.suspect_data["suspect_id"])
        if self.suspect_data["suspect_time"] > cfg.REID.SUSPECT_TIME:
            self.suspect_data["major"] = majority_element(self.suspect_data["suspect_id"])
            self.suspect = False
    
    def update_major(self):
        self.suspect_data["major"] = majority_element(self.suspect_data["suspect_id"])
    
    def update(self, det):
        """
        Updates the state vector with observed bbox.
        """

        if det is not None:
            bbox = det[0:5]
            self.conf = det[4]
            self.cls = det[5]
            self.det_ind = det[6]
            self.frozen = False

            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            if self.new_kf:
                R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
                self.kf.update(self.bbox_to_z_func(bbox), R=R)
            else:
                self.kf.update(self.bbox_to_z_func(bbox))
        else:
            self.kf.update(det)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        # For OCR
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        # Apply to each box in the range of velocity computation
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        # Also need to change kf state, but might be frozen
        self.kf.apply_affine_correction(m, t, self.new_kf)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Don't allow negative bounding boxes
        if self.new_kf:
            if self.kf.x[2] + self.kf.x[6] <= 0:
                self.kf.x[6] = 0
            if self.kf.x[3] + self.kf.x[7] <= 0:
                self.kf.x[7] = 0

            # Stop velocity, will update in kf during OOS
            if self.frozen:
                self.kf.x[6] = self.kf.x[7] = 0
            Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
        else:
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0
            Q = None

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))

