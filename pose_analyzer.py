import mediapipe as mp
import cv2
import numpy as np
from collections import deque

class PoseAnalyzer:
    def __init__(self, smoothing_window=3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.elbow_smooth = deque(maxlen=5)

        # Landmark smoothing — store as plain floats, not protobuf objects
        self._smooth_window = smoothing_window
        self._lm_history = deque(maxlen=smoothing_window)
        self._last_smoothed = None  # list of dicts with x,y,z,visibility

    def _landmarks_to_dicts(self, landmarks):
        return [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                for lm in landmarks]

    def _smooth_landmarks(self, current_dicts):
        self._lm_history.append(current_dicts)
        if len(self._lm_history) == 1:
            self._last_smoothed = current_dicts
            return current_dicts

        n = len(self._lm_history)
        smoothed = []
        for i in range(len(current_dicts)):
            sx = sum(h[i]['x'] for h in self._lm_history) / n
            sy = sum(h[i]['y'] for h in self._lm_history) / n
            sz = sum(h[i]['z'] for h in self._lm_history) / n
            sv = min(h[i]['visibility'] for h in self._lm_history)
            smoothed.append({'x': sx, 'y': sy, 'z': sz, 'visibility': sv})
        self._last_smoothed = smoothed
        return smoothed

    def _draw_smoothed(self, image, smoothed, connections):
        h, w, _ = image.shape
        for conn in connections:
            idx0, idx1 = conn
            p0 = smoothed[idx0]
            p1 = smoothed[idx1]
            if p0['visibility'] < 0.3 or p1['visibility'] < 0.3:
                continue
            x0, y0 = int(p0['x'] * w), int(p0['y'] * h)
            x1, y1 = int(p1['x'] * w), int(p1['y'] * h)
            cv2.line(image, (x0, y0), (x1, y1), (0, 0, 255), 2)

        for lm in smoothed:
            if lm['visibility'] < 0.3:
                continue
            cx, cy = int(lm['x'] * w), int(lm['y'] * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)

    def calculate_angle(self, point1, point2, point3):
        # Accepts both protobuf landmarks and dicts
        def coords(p):
            if isinstance(p, dict):
                return np.array([p['x'], p['y']])
            return np.array([p.x, p.y])
        a = coords(point1)
        b = coords(point2)
        c = coords(point3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def landmarks_visible(self, landmarks, indices, threshold=0.3):
        for i in indices:
            lm = landmarks[i]
            vis = lm['visibility'] if isinstance(lm, dict) else lm.visibility
            if vis <= threshold:
                return False
        return True

    def _vis(self, lm):
        return lm['visibility'] if isinstance(lm, dict) else lm.visibility

    def _x(self, lm):
        return lm['x'] if isinstance(lm, dict) else lm.x

    def _y(self, lm):
        return lm['y'] if isinstance(lm, dict) else lm.y

    def best_side(self, landmarks):
        PL = self.mp_pose.PoseLandmark
        r_vis = self._vis(landmarks[PL.RIGHT_ELBOW.value])
        l_vis = self._vis(landmarks[PL.LEFT_ELBOW.value])
        if r_vis >= l_vis:
            return (PL.RIGHT_SHOULDER.value, PL.RIGHT_ELBOW.value, PL.RIGHT_WRIST.value,
                    PL.RIGHT_HIP.value, PL.RIGHT_KNEE.value, PL.RIGHT_ANKLE.value)
        else:
            return (PL.LEFT_SHOULDER.value, PL.LEFT_ELBOW.value, PL.LEFT_WRIST.value,
                    PL.LEFT_HIP.value, PL.LEFT_KNEE.value, PL.LEFT_ANKLE.value)

    def calculate_6_angles(self, landmarks):
        PL = self.mp_pose.PoseLandmark
        sh_idx, el_idx, wr_idx, hip_idx, kn_idx, an_idx = self.best_side(landmarks)

        shoulder = landmarks[sh_idx]
        elbow    = landmarks[el_idx]
        wrist    = landmarks[wr_idx]
        hip      = landmarks[hip_idx]
        knee     = landmarks[kn_idx]
        ankle    = landmarks[an_idx]

        angles = {}

        elbow_ids = [sh_idx, el_idx, wr_idx]
        if self.landmarks_visible(landmarks, elbow_ids):
            raw_elbow = self.calculate_angle(shoulder, elbow, wrist)
            self.elbow_smooth.append(raw_elbow)
            angles['elbow'] = np.mean(self.elbow_smooth)
        elif self.elbow_smooth:
            angles['elbow'] = np.mean(self.elbow_smooth)
        else:
            angles['elbow'] = 180.0

        flare_ids = [hip_idx, sh_idx, el_idx]
        if self.landmarks_visible(landmarks, flare_ids):
            angles['elbow_flare'] = self.calculate_angle(hip, shoulder, elbow)
        else:
            angles['elbow_flare'] = 45.0

        hip_ids = [sh_idx, hip_idx, kn_idx]
        if self.landmarks_visible(landmarks, hip_ids):
            angles['hip'] = self.calculate_angle(shoulder, hip, knee)
        else:
            angles['hip'] = 170.0

        wrist_ids = [el_idx, wr_idx]
        if self.landmarks_visible(landmarks, wrist_ids):
            ex, ey = self._x(elbow), self._y(elbow)
            wx, wy = self._x(wrist), self._y(wrist)
            vpx = wx + (wx - ex) * 0.2
            vpy = wy + (wy - ey) * 0.2
            vp = {'x': vpx, 'y': vpy}
            angles['wrist'] = self.calculate_angle(elbow, wrist, vp)
        else:
            angles['wrist'] = 170.0

        r_sh = landmarks[PL.RIGHT_SHOULDER.value]
        l_sh = landmarks[PL.LEFT_SHOULDER.value]
        r_vis = self._vis(r_sh) > 0.3
        l_vis = self._vis(l_sh) > 0.3
        if r_vis and l_vis:
            angles['shoulder_elevation'] = abs(self._y(r_sh) - self._y(l_sh)) * 100
        elif r_vis or l_vis:
            angles['shoulder_elevation'] = abs(self._y(shoulder) - self._y(hip)) * 10
        else:
            angles['shoulder_elevation'] = 0.0

        body_ids = [an_idx, hip_idx, sh_idx]
        if self.landmarks_visible(landmarks, body_ids):
            angles['body_line'] = abs(180 - self.calculate_angle(ankle, hip, shoulder))
        else:
            angles['body_line'] = 0.0

        return angles

    def get_angle_status(self, angle_name, angle_value):
        ranges = {
            'elbow':              {'good': (80, 180),  'warning': (70, 80)},
            'elbow_flare':        {'good': (35, 50),   'warning': (30, 60)},
            'hip':                {'good': (160, 175), 'warning': (150, 160)},
            'wrist':              {'good': (160, 180), 'warning': (150, 160)},
            'shoulder_elevation': {'good': (0, 3),     'warning': (3, 6)},
            'body_line':          {'good': (0, 10),    'warning': (10, 20)}
        }
        if angle_name not in ranges:
            return 'unknown'
        r = ranges[angle_name]
        if r['good'][0] <= angle_value <= r['good'][1]:
            return 'good'
        elif r['warning'][0] <= angle_value <= r['warning'][1]:
            return 'warning'
        else:
            return 'bad'

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        angles = None
        landmarks = None

        if results.pose_landmarks:
            raw_lm = results.pose_landmarks.landmark
            lm_dicts = self._landmarks_to_dicts(raw_lm)
            smoothed = self._smooth_landmarks(lm_dicts)

            self._draw_smoothed(image, smoothed, self.mp_pose.POSE_CONNECTIONS)

            landmarks = smoothed
            angles = self.calculate_6_angles(smoothed)

            y_offset = 30
            for angle_name, angle_value in angles.items():
                status = self.get_angle_status(angle_name, angle_value)
                color = (0, 255, 0) if status == 'good' else (0, 165, 255) if status == 'warning' else (0, 0, 255)
                display_name = angle_name.replace('_', ' ').title()
                cv2.putText(image, f"{display_name}: {angle_value:.1f}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30

        return image, angles, landmarks

    def get_last_smoothed(self):
        return self._last_smoothed