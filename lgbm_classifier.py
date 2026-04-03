import pickle
import numpy as np
from collections import deque
from user_calibration import UserCalibration


class LGBMPushupClassifier:
    def __init__(self, model_path='rehabmate_v3.pkl',
                 calibration_path='user_calibration.json'):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.lgbm_model = data['lgbm']
        self.rf_model   = data['rf']
        self.le         = data['encoder']
        self.features   = data['features']
        self.weights    = data.get('weights', [3, 2])
        self.model      = self.lgbm_model

        self.elbow_hist = deque(maxlen=10)
        self.hip_hist   = deque(maxlen=10)
        self.vel_hist   = deque(maxlen=10)

        # Cumulative rep tracking (resets each rep)
        self._rep_elbow_frames = []
        self._rep_hip_frames   = []
        self._rep_all_frames   = []  # store full feature dicts for rep-level aggregation

        self.calibration = UserCalibration()
        self.calibration.load(calibration_path)
        self.calibration_path = calibration_path

    def _velocity(self, history):
        if len(history) < 2:
            return 0.0
        hist = list(history)
        return float(np.mean([(hist[i] - hist[i-1]) * 30 for i in range(1, len(hist))]))

    def add_calibration_sample(self, angles: dict):
        self.calibration.add_sample(angles)

    def finalize_calibration(self, save_path='user_calibration.json') -> bool:
        success = self.calibration.compute_offsets(min_samples=10)
        if success:
            self.calibration.save(save_path)
        return success

    def update_rep(self, rep_bottom_elbow, rep_bottom_hip):
        self._rep_elbow_frames.clear()
        self._rep_hip_frames.clear()
        self._rep_all_frames.clear()

    def reset(self):
        self.elbow_hist.clear()
        self.hip_hist.clear()
        self.vel_hist.clear()
        self._rep_elbow_frames.clear()
        self._rep_hip_frames.clear()
        self._rep_all_frames.clear()

    def predict(self, angles, phase, state_duration, rep_number, baseline):
        elbow = angles['elbow']
        hip   = angles['hip']

        self.elbow_hist.append(elbow)
        self.hip_hist.append(hip)
        self._rep_elbow_frames.append(elbow)
        self._rep_hip_frames.append(hip)

        elbow_vel = self._velocity(self.elbow_hist)
        hip_vel   = self._velocity(self.hip_hist)
        self.vel_hist.append(elbow_vel)
        vel_trend = float(np.mean(self.vel_hist)) if self.vel_hist else 0.0

        phase_key = 'down' if phase == 'down' else 'up' if phase == 'up' else 'mid'
        elbow_ref = baseline.get(f'elbow_{phase_key}', 90)
        hip_ref   = baseline.get(f'hip_{phase_key}', 170)

        cal = self.calibration.apply_dict(angles)

        # Cumulative rep stats (only past + current frame, no future)
        ef = self._rep_elbow_frames
        hf = self._rep_hip_frames
        e_cum_min = min(ef)
        e_cum_max = max(ef)
        h_cum_min = min(hf)
        h_cum_max = max(hf)

        x = np.array([[
            cal.get('elbow', elbow),
            cal.get('elbow_flare', angles['elbow_flare']),
            cal.get('hip', hip),
            cal.get('shoulder_elevation', angles['shoulder_elevation']),
            cal.get('body_line', angles['body_line']),
            elbow_vel,
            hip_vel,
            state_duration,
            elbow - elbow_ref,
            hip - hip_ref,
            vel_trend,
            elbow - (float(np.mean(list(self.elbow_hist))) if self.elbow_hist else elbow),
            hip - (float(np.mean(list(self.hip_hist))) if self.hip_hist else hip),
            hip / (elbow + 1e-6),
            angles['body_line'] - (180 - hip),
            e_cum_min,
            e_cum_max,
            e_cum_max - e_cum_min,
            h_cum_min,
            h_cum_max,
            h_cum_max - h_cum_min,
        ]])

        # Store for rep-level aggregation
        self._rep_all_frames.append(x[0].copy())

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lgbm_p = self.lgbm_model.predict_proba(x)[0]
            rf_p   = self.rf_model.predict_proba(x)[0]

        w = self.weights
        ens_p = (w[0] * lgbm_p + w[1] * rf_p) / sum(w)

        pred_idx   = int(np.argmax(ens_p))
        label      = self.le.inverse_transform([pred_idx])[0]
        confidence = float(ens_p[pred_idx])

        return label, confidence, dict(zip(self.le.classes_, ens_p))

    def predict_rep(self):
        """Aggregate all frames in this rep and predict once."""
        if len(self._rep_all_frames) == 0:
            return 'perfect', 0.0, {}

        frames = np.array(self._rep_all_frames)
        x_avg = frames.mean(axis=0).reshape(1, -1)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lgbm_p = self.lgbm_model.predict_proba(x_avg)[0]
            rf_p   = self.rf_model.predict_proba(x_avg)[0]

        w = self.weights
        ens_p = (w[0] * lgbm_p + w[1] * rf_p) / sum(w)

        pred_idx   = int(np.argmax(ens_p))
        label      = self.le.inverse_transform([pred_idx])[0]
        confidence = float(ens_p[pred_idx])

        return label, confidence, dict(zip(self.le.classes_, ens_p))

    @property
    def is_calibrated(self) -> bool:
        return self.calibration.calibrated

    @staticmethod
    def get_feedback(label):
        return {
            'perfect':         'Great form!',
            'hip_sag':         'Hips dropping - engage core',
            'fatigue_hip_sag': 'Fatigue - rest or slow down',
            'elbow_flare':     'Elbows flaring - tuck them in',
            'shallow':         'Go deeper - chest to floor',
            'shoulder_dip':    'Shoulders uneven - keep level',
        }.get(label, '')

    @staticmethod
    def get_color(label):
        return {
            'perfect':         (0, 220, 0),
            'hip_sag':         (0, 165, 255),
            'fatigue_hip_sag': (0, 0, 255),
            'elbow_flare':     (0, 165, 255),
            'shallow':         (0, 165, 255),
            'shoulder_dip':    (0, 165, 255),
        }.get(label, (255, 255, 255))