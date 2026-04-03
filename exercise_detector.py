import numpy as np

class ExerciseDetector:
    def __init__(self):
        self.position_confidence = 0.0
        self.required_confidence = 0.7

    def is_pushup_position(self, angles, landmarks=None):
        if not angles:
            return False, 0.0, "No pose detected"

        checks  = []
        feedback = []

        if landmarks is not None:
            hip_indices   = [23, 24]
            ankle_indices = [27, 28]

            def vis(idx):
                lm = landmarks[idx]
                return lm['visibility'] if isinstance(lm, dict) else lm.visibility

            hip_visible   = any(vis(i) > 0.4 for i in hip_indices)
            ankle_visible = any(vis(i) > 0.4 for i in ankle_indices)

            if not hip_visible or not ankle_visible:
                self.position_confidence = 0.0
                return False, 0.0, "GET IN POSITION - Full body not visible"

        hip_angle = angles.get('hip', 0)
        if 120 <= hip_angle <= 185:
            checks.append(1.0)
        elif 100 <= hip_angle < 120:
            checks.append(0.5)
            feedback.append("Lower your hips a little more")
        else:
            checks.append(0.0)
            feedback.append("Bend forward into pushup position" if hip_angle < 100 else "Lower your body")

        elbow_angle = angles.get('elbow', 0)
        if 60 <= elbow_angle <= 180:
            checks.append(1.0)
        else:
            checks.append(0.0)
            feedback.append("Position arms for pushup")

        body_line = angles.get('body_line', 0)
        if body_line < 20:
            checks.append(1.0)
        elif body_line < 30:
            checks.append(0.5)
            feedback.append("Lower your hips to horizontal")
        else:
            checks.append(0.0)
            feedback.append("Body too vertical - get horizontal")

        # Reject fallback values
        if body_line == 0.0 and hip_angle >= 168:
            self.position_confidence = 0.0
            return False, 0.0, "GET IN POSITION - Move back so full body is in frame"

        confidence = sum(checks) / len(checks)
        if confidence >= 0.7:
            self.position_confidence = (self.position_confidence * 0.7) + (confidence * 0.3)
        else:
            self.position_confidence = (self.position_confidence * 0.3) + (confidence * 0.7)

        is_in_position = self.position_confidence >= self.required_confidence

        if is_in_position:
            message = "READY - In position!"
        elif confidence > 0.5:
            message = "ALMOST - " + (feedback[-1] if feedback else "Adjust position")
        else:
            message = "GET IN POSITION - " + (feedback[0] if feedback else "Start plank")

        return is_in_position, self.position_confidence, message