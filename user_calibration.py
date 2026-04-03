import numpy as np
import json
import os

CALIBRATABLE_FEATURES = ['elbow', 'hip', 'body_line', 'elbow_flare', 'shoulder_elevation']

POPULATION_MEANS = {
    'elbow':             119.23,
    'hip':               168.47,
    'shoulder_elevation':  2.60,
    'elbow_flare':        21.25,
    'body_line':          12.55,
}

FEATURES = [
    'elbow', 'elbow_flare', 'hip', 'shoulder_elevation', 'body_line',
    'elbow_velocity', 'hip_velocity', 'time_in_phase', 'rep_number',
    'elbow_drift', 'hip_drift', 'velocity_trend', 'rolling_elbow_5', 'rolling_hip_5'
]

CALIBRATION_FILE = 'user_calibration.json'


class UserCalibration:
    def __init__(self):
        self.offsets      = {f: 0.0 for f in CALIBRATABLE_FEATURES}
        self.user_means   = {}
        self.n_samples    = 0
        self.calibrated   = False
        self._buffer      = {f: [] for f in CALIBRATABLE_FEATURES}

    def add_sample(self, feature_dict: dict):
        for f in CALIBRATABLE_FEATURES:
            if f in feature_dict:
                self._buffer[f].append(float(feature_dict[f]))
        self.n_samples += 1

    # Alias for collect_data.py compatibility
    def add_calibration_sample(self, feature_dict: dict):
        self.add_sample(feature_dict)

    def compute_offsets(self, min_samples: int = 20) -> bool:
        if self.n_samples < min_samples:
            print(f"[Calibration] Need {min_samples} samples, only have {self.n_samples}")
            return False

        for f in CALIBRATABLE_FEATURES:
            if len(self._buffer[f]) == 0:
                continue
            user_mean       = np.mean(self._buffer[f])
            self.user_means[f] = user_mean
            self.offsets[f] = POPULATION_MEANS[f] - user_mean

        self.calibrated = True

        print(f"[Calibration] Computed from {self.n_samples} samples")
        for f in CALIBRATABLE_FEATURES:
            print(f"  {f:<22} user={self.user_means.get(f, 0):>8.2f}  pop={POPULATION_MEANS[f]:>8.2f}  offset={self.offsets[f]:>+8.2f}")
        return True

    # Alias for collect_data.py compatibility
    def finalize_calibration(self, save_path=CALIBRATION_FILE, min_samples=10):
        ok = self.compute_offsets(min_samples=min_samples)
        if ok:
            self.save(save_path)
        return ok

    def apply(self, feature_vector: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            return feature_vector

        corrected = feature_vector.copy().astype(float)
        is_1d = corrected.ndim == 1

        if is_1d:
            corrected = corrected.reshape(1, -1)

        for f, offset in self.offsets.items():
            if offset == 0.0:
                continue
            idx = FEATURES.index(f)
            corrected[:, idx] += offset

        return corrected[0] if is_1d else corrected

    def apply_dict(self, feature_dict: dict) -> dict:
        if not self.calibrated:
            return feature_dict

        corrected = dict(feature_dict)
        for f, offset in self.offsets.items():
            if f in corrected and offset != 0.0:
                corrected[f] = float(corrected[f]) + offset
        return corrected

    def save(self, path: str = CALIBRATION_FILE):
        data = {
            'calibrated':   self.calibrated,
            'n_samples':    self.n_samples,
            'offsets':      self.offsets,
            'user_means':   self.user_means,
            'population_means': POPULATION_MEANS,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Calibration] Saved to {path}")

    def load(self, path: str = CALIBRATION_FILE) -> bool:
        if not os.path.exists(path):
            print(f"[Calibration] No file at {path}")
            return False
        with open(path) as f:
            data = json.load(f)
        self.calibrated  = data['calibrated']
        self.n_samples   = data['n_samples']
        self.offsets     = data['offsets']
        self.user_means  = data['user_means']
        print(f"[Calibration] Loaded from {path} ({self.n_samples} samples)")
        return self.calibrated

    def reset(self):
        self.__init__()
        print("[Calibration] Reset")

    @property
    def status(self) -> dict:
        return {
            'calibrated': self.calibrated,
            'n_samples':  self.n_samples,
            'offsets':    self.offsets,
        }