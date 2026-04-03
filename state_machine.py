import time
import numpy as np
from collections import deque

class PushupStateMachine:
    def __init__(self):
        self.state = "READY"
        self.rep_count = 0
        self.valid_rep_count = 0
        self.state_start_time = time.time()
        self.state_duration = 0

        self.angle_history    = deque(maxlen=10)
        self.current_rep_depth   = 180
        self.current_rep_hip_min = 180
        self.last_rep_quality    = "N/A"

        self._top_confirm_frames      = 0
        self._descend_confirm_frames  = 0
        self._bottom_confirm_frames   = 0
        self._reached_bottom          = False
        self._not_in_pos_frames       = 0

        # Relaxed defaults — catches more reps, calibrate() refines them
        self.BOTTOM_THRESHOLD  = 120
        self.TOP_THRESHOLD     = 140
        self.MIN_GOOD_DEPTH    = 95
        self.HIP_SAG_THRESHOLD = 130
        self.CONFIRM_FRAMES    = 2
        self.BAIL_FRAMES       = 8

    def calibrate(self, calibration_profile: dict):
        elbow_bottom = calibration_profile.get('elbow_bottom', 70.0)
        elbow_top    = calibration_profile.get('elbow_top',   165.0)
        self.BOTTOM_THRESHOLD = int(elbow_bottom + 30)
        self.TOP_THRESHOLD    = int(elbow_top - 25)
        self.MIN_GOOD_DEPTH   = int(elbow_bottom + 15)
        print(f"[StateMachine] Calibrated — "
              f"BOTTOM<{self.BOTTOM_THRESHOLD} "
              f"TOP>{self.TOP_THRESHOLD} "
              f"GOOD<{self.MIN_GOOD_DEPTH}")

    def get_elbow_velocity(self):
        if len(self.angle_history) < 2:
            return 0
        hist = list(self.angle_history)
        return sum([(hist[i] - hist[i-1]) * 30
                    for i in range(1, len(hist))]) / (len(hist) - 1)

    def update(self, angles, is_in_position):
        if not angles:
            return self.get_state_info()

        elbow_angle = angles.get('elbow', 180)
        hip_angle   = angles.get('hip', 170)
        hip_valid = hip_angle != 170.0

        if not is_in_position:
            self._not_in_pos_frames += 1
            if self.state in ("READY", "REST"):
                return self.get_state_info()
            if self._not_in_pos_frames >= self.BAIL_FRAMES:
                self._abort_rep()
                return self.get_state_info()
        else:
            self._not_in_pos_frames = 0

        self.angle_history.append(elbow_angle)
        velocity        = self.get_elbow_velocity()
        self.state_duration = time.time() - self.state_start_time

        if elbow_angle < self.current_rep_depth:
            self.current_rep_depth = elbow_angle
        if hip_valid and hip_angle < self.current_rep_hip_min:
            self.current_rep_hip_min = hip_angle

        if self.state == "READY":
            if elbow_angle < 160 and velocity < -5:
                self._descend_confirm_frames += 1
                if self._descend_confirm_frames >= self.CONFIRM_FRAMES:
                    self._descend_confirm_frames = 0
                    self._reached_bottom = False
                    self.transition_to("DESCENDING")
            else:
                self._descend_confirm_frames = 0

        elif self.state == "DESCENDING":
            if elbow_angle < self.BOTTOM_THRESHOLD:
                self._bottom_confirm_frames += 1
                if self._bottom_confirm_frames >= self.CONFIRM_FRAMES:
                    self._bottom_confirm_frames = 0
                    self._reached_bottom = True
                    self.transition_to("BOTTOM")
            else:
                self._bottom_confirm_frames = 0

            if velocity > 15 and elbow_angle > 150:
                self._reached_bottom = False
                self.transition_to("READY")

        elif self.state == "BOTTOM":
            if self.state_duration > 0.05 and velocity > 3:
                self.transition_to("ASCENDING")

        elif self.state == "ASCENDING":
            if elbow_angle > self.TOP_THRESHOLD:
                self._top_confirm_frames += 1
                if self._top_confirm_frames >= self.CONFIRM_FRAMES:
                    self._top_confirm_frames = 0
                    if self._reached_bottom:
                        self.complete_rep()
                    self._reached_bottom = False
                    self.transition_to("TOP")
            else:
                self._top_confirm_frames = 0

        elif self.state == "TOP":
            if self.state_duration > 0.15:
                if velocity < -5:
                    self.transition_to("DESCENDING")
                elif self.state_duration > 3:
                    self.transition_to("REST")

        elif self.state == "REST":
            if velocity < -5:
                self.transition_to("DESCENDING")

        return self.get_state_info()

    def _abort_rep(self):
        self._reached_bottom = False
        self.current_rep_depth   = 180
        self.current_rep_hip_min = 180
        self._top_confirm_frames     = 0
        self._descend_confirm_frames = 0
        self._bottom_confirm_frames  = 0
        self._not_in_pos_frames      = 0
        self.transition_to("READY")

    def transition_to(self, new_state):
        self.state           = new_state
        self.state_start_time = time.time()
        self.state_duration  = 0

    def complete_rep(self):
        quality = self.evaluate_rep_quality()
        self.rep_count += 1
        if quality in ["EXCELLENT", "GOOD", "SHALLOW"]:
            self.valid_rep_count += 1
        self.last_rep_quality    = quality
        self.current_rep_depth   = 180
        self.current_rep_hip_min = 180

    def evaluate_rep_quality(self):
        if self.current_rep_depth > 130:
            return "SHALLOW"
        if self.current_rep_hip_min < self.HIP_SAG_THRESHOLD:
            return "HIP_SAG"
        if self.current_rep_depth < self.MIN_GOOD_DEPTH:
            return "EXCELLENT"
        return "GOOD"

    def get_state_info(self):
        return {
            'state':            self.state,
            'rep_count':        self.rep_count,
            'valid_rep_count':  self.valid_rep_count,
            'state_duration':   self.state_duration,
            'last_rep_quality': self.last_rep_quality,
            'current_depth':    self.current_rep_depth
        }

    def reset(self):
        self.rep_count           = 0
        self.valid_rep_count     = 0
        self.state               = "READY"
        self.last_rep_quality    = "N/A"
        self.current_rep_depth   = 180
        self.current_rep_hip_min = 180
        self._top_confirm_frames     = 0
        self._descend_confirm_frames = 0
        self._bottom_confirm_frames  = 0
        self._reached_bottom         = False
        self._not_in_pos_frames      = 0