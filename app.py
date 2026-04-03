import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import os
import time
import json as _json
from collections import deque, defaultdict

from pose_analyzer import PoseAnalyzer
from exercise_detector import ExerciseDetector
from state_machine import PushupStateMachine
from lgbm_classifier import LGBMPushupClassifier
from user_calibration import UserCalibration

app = Flask(__name__)

CALIB_REPS_REQUIRED = 5
MODEL_PATH          = 'rehabmate_v3.pkl'
CALIB_JSON          = 'user_calibration.json'
CALIB_PROFILE       = 'calibration_profile.json'
TOLERANCE           = 10.0
HIP_TOLERANCE       = 15.0

STATE_PHASE = {
    'TOP': 'up', 'ASCENDING': 'up',
    'BOTTOM': 'down', 'DESCENDING': 'mid', 'READY': 'mid'
}

def load_baseline():
    if os.path.exists(CALIB_PROFILE):
        with open(CALIB_PROFILE) as f:
            _c = _json.load(f)
        return {
            'elbow_top':     _c.get('elbow_top',    165.0),
            'elbow_bottom':  _c.get('elbow_bottom',  70.0),
            'elbow_mid':     _c.get('elbow_mid',    120.0),
            'hip_top':       _c.get('hip_top',      168.0),
            'hip_bottom':    _c.get('hip_bottom',   158.0),
            'hip_mid':       _c.get('hip_mid',      168.0),
            'body_line':     _c.get('body_line',     10.0),
            'velocity_desc': _c.get('velocity_desc',  0.0),
            'tolerance':     _c.get('tolerance',     10.0),
            'hip_tolerance': _c.get('hip_tolerance', 15.0),
        }
    return {
        'elbow_top': 167.0, 'elbow_bottom': 70.0, 'elbow_mid': 120.0,
        'hip_top': 164.0,   'hip_bottom': 130.0,   'hip_mid': 147.0,
        'body_line': 10.0,  'velocity_desc': 0.0,
        'tolerance': 10.0,  'hip_tolerance': 15.0
    }

BASELINE = load_baseline()

analyzer          = PoseAnalyzer()
exercise_detector = ExerciseDetector()
state_machine     = PushupStateMachine()
if os.path.exists(CALIB_PROFILE):
    state_machine.calibrate(BASELINE)

classifier = LGBMPushupClassifier(model_path=MODEL_PATH, calibration_path=CALIB_JSON)

user_calib = UserCalibration()
if os.path.exists(CALIB_JSON):
    user_calib.load(CALIB_JSON)

# Calibration state
calib_active     = not os.path.exists(CALIB_PROFILE)
calib_sm         = PushupStateMachine()
calib_rep_data   = defaultdict(lambda: defaultdict(list))
calib_last_rep   = 0
calib_elbow_hist = deque(maxlen=10)
calib_hip_hist   = deque(maxlen=10)
calib_message    = "Do 5 slow perfect pushups to calibrate"

calibration_status = {
    'done': not calib_active,
    'reps_done': 0 if calib_active else CALIB_REPS_REQUIRED,
    'reps_required': CALIB_REPS_REQUIRED,
    'message': '' if not calib_active else 'Calibration needed'
}

camera_lock = threading.Lock()
camera      = None

current_angles          = {}
current_position_status = {'in_position': False, 'confidence': 0.0, 'message': 'Waiting...'}
current_state_info      = {
    'state': 'READY', 'rep_count': 0, 'valid_rep_count': 0,
    'state_duration': 0, 'last_rep_quality': 'N/A'
}
current_form = {
    'label': 'N/A', 'confidence': 0.0, 'feedback': '',
    'color': [255, 255, 255], 'all_probs': {}
}

rep_history     = []
last_rep_seen   = 0
beep_pending    = False

output_frame = None
frame_lock   = threading.Lock()

latest_raw_frame = None
raw_frame_lock   = threading.Lock()


def compute_velocity(history):
    if len(history) < 2:
        return 0.0
    hist = list(history)
    return float(sum([(hist[i] - hist[i-1]) * 30 for i in range(1, len(hist))]) / (len(hist) - 1))


def finalize_calibration():
    global BASELINE, calib_active, calib_message, calibration_status

    valid_reps = [r for r in calib_rep_data if r >= 2]
    if not valid_reps:
        calib_message = "Calibration failed — redo"
        return

    def avg(key):
        vals = []
        for r in valid_reps:
            vals.extend(calib_rep_data[r].get(key, []))
        return float(sum(vals)/len(vals)) if vals else None

    baseline = {
        'elbow_top':     avg('up_elbow'),
        'elbow_bottom':  avg('down_elbow'),
        'elbow_mid':     avg('mid_elbow'),
        'hip_top':       avg('up_hip'),
        'hip_bottom':    avg('down_hip'),
        'hip_mid':       avg('mid_hip'),
        'body_line':     avg('up_body'),
        'velocity_desc': avg('mid_vel'),
        'tolerance':     TOLERANCE,
        'hip_tolerance': HIP_TOLERANCE,
        'reps_used':     valid_reps,
    }

    with open(CALIB_PROFILE, 'w') as f:
        _json.dump(baseline, f, indent=2)

    BASELINE = baseline
    state_machine.calibrate(baseline)

    # Also compute user calibration offsets
    user_calib.compute_offsets(min_samples=5)
    user_calib.save(CALIB_JSON)
    classifier.calibration_path = CALIB_JSON

    calib_active = False
    calibration_status['done'] = True
    calibration_status['reps_done'] = CALIB_REPS_REQUIRED
    calibration_status['message'] = 'Calibration complete'
    calib_message = "Calibration done!"
    print(f"[Calibration] Saved — elbow top={baseline['elbow_top']:.1f} bottom={baseline['elbow_bottom']:.1f}")


def camera_reader():
    global latest_raw_frame, camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    with camera_lock:
        camera = cam

    while True:
        ret = cam.grab()
        if not ret:
            time.sleep(0.001)
            continue
        ret, frame = cam.retrieve()
        if ret:
            frame = cv2.flip(frame, 1)
            with raw_frame_lock:
                latest_raw_frame = frame


def process_calibration_frame(frame):
    global calib_last_rep, calib_message

    annotated, angles, landmarks = analyzer.process_frame(frame)
    h, w, _ = annotated.shape

    if angles:
        is_in_pos, conf, msg = exercise_detector.is_pushup_position(angles, landmarks)
        info = calib_sm.update(angles, is_in_pos)
        state = info['state']
        rep   = info['rep_count']

        calib_elbow_hist.append(angles['elbow'])
        calib_hip_hist.append(angles['hip'])

        if rep >= 2:
            user_calib.add_sample(angles)

        phase = STATE_PHASE.get(state)
        if phase and rep > 0:
            calib_rep_data[rep][phase + '_elbow'].append(angles['elbow'])
            calib_rep_data[rep][phase + '_hip'].append(angles['hip'])
            calib_rep_data[rep][phase + '_vel'].append(compute_velocity(calib_elbow_hist))
            calib_rep_data[rep][phase + '_body'].append(angles['body_line'])

        collected = len(calib_rep_data)
        calibration_status['reps_done'] = collected

        # Overlay
        ov = annotated.copy()
        cv2.rectangle(ov, (0, 0), (w, 70), (0, 100, 200), -1)
        annotated = cv2.addWeighted(annotated, 0.5, ov, 0.5, 0)
        cv2.putText(annotated, "CALIBRATION", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(annotated, f"Do {CALIB_REPS_REQUIRED} slow perfect reps  |  {collected}/{CALIB_REPS_REQUIRED}",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        ov2 = annotated.copy()
        cv2.rectangle(ov2, (0, h-40), (w, h), (30, 30, 30), -1)
        annotated = cv2.addWeighted(annotated, 0.65, ov2, 0.35, 0)
        cv2.putText(annotated, f"State: {state}  Rep: {rep}",
                    (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if collected >= CALIB_REPS_REQUIRED:
            finalize_calibration()

    return annotated


def process_loop():
    global output_frame
    global current_angles, current_position_status, current_state_info, current_form
    global rep_history, last_rep_seen, beep_pending

    while True:
        with raw_frame_lock:
            f = latest_raw_frame
        if f is not None:
            break
        time.sleep(0.01)

    last_annotated = None
    process_interval = 1.0 / 12
    last_process_time = 0

    while True:
        try:
            with raw_frame_lock:
                frame = latest_raw_frame

            if frame is None:
                time.sleep(0.005)
                continue

            now = time.time()

            if now - last_process_time < process_interval:
                if last_annotated is not None:
                    ret2, buf = cv2.imencode('.jpg', last_annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if ret2:
                        with frame_lock:
                            output_frame = buf.tobytes()
                time.sleep(0.005)
                continue

            last_process_time = now

            # Calibration phase
            if calib_active:
                annotated = process_calibration_frame(frame)
                last_annotated = annotated
                ret2, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret2:
                    with frame_lock:
                        output_frame = buf.tobytes()
                continue

            # Normal processing
            annotated, angles, landmarks = analyzer.process_frame(frame)

            if angles:
                current_angles = angles
                is_in_pos, conf, msg = exercise_detector.is_pushup_position(angles, landmarks)
                current_position_status = {'in_position': is_in_pos, 'confidence': conf, 'message': msg}
                h, w, _ = annotated.shape

                si = state_machine.update(angles, is_in_pos)
                current_state_info = si
                phase   = STATE_PHASE.get(si['state'], 'mid')
                rep_num = si.get('rep_count', 0)
                dur     = si.get('state_duration', 0)

                # Clear frame buffer if position lost or state reset
                if not is_in_pos or si['state'] == 'READY':
                    if rep_num == last_rep_seen:
                        classifier._rep_elbow_frames.clear()
                        classifier._rep_hip_frames.clear()
                        classifier._rep_all_frames.clear()

                # Accumulate frames during rep phases (predict() does this internally)
                if is_in_pos and si['state'] in ('DESCENDING', 'BOTTOM', 'ASCENDING'):
                    classifier.predict(angles, phase, dur, rep_num, BASELINE)

                # Rep completed — run rep-level prediction from averaged frames
                if rep_num > last_rep_seen and len(classifier._rep_all_frames) >= 2:
                    last_rep_seen = rep_num
                    beep_pending  = True

                    best_label, best_conf, all_probs = classifier.predict_rep()
                    n = len(classifier._rep_all_frames)

                    # Angle averages for display
                    e_frames = classifier._rep_elbow_frames
                    h_frames = classifier._rep_hip_frames
                    angle_avg = {
                        'elbow': round(sum(e_frames)/len(e_frames), 1) if e_frames else 0,
                        'hip':   round(sum(h_frames)/len(h_frames), 1) if h_frames else 0,
                        'elbow_flare': round(angles.get('elbow_flare', 0), 1),
                        'shoulder_elevation': round(angles.get('shoulder_elevation', 0), 1),
                        'body_line': round(angles.get('body_line', 0), 1),
                    }

                    classifier.update_rep(0, 0)

                    current_form = {
                        'label':      best_label,
                        'confidence': round(best_conf, 3),
                        'feedback':   classifier.get_feedback(best_label),
                        'color':      list(classifier.get_color(best_label)),
                        'all_probs':  {k: round(float(v), 3) for k, v in all_probs.items()}
                    }
                    rep_history.append({
                        'rep':                rep_num,
                        'label':              best_label,
                        'confidence':         round(best_conf * 100, 1),
                        'quality':            si['last_rep_quality'],
                        'elbow':              angle_avg.get('elbow', 0),
                        'hip':                angle_avg.get('hip', 0),
                        'elbow_flare':        angle_avg.get('elbow_flare', 0),
                        'shoulder_elevation': angle_avg.get('shoulder_elevation', 0),
                        'body_line':          angle_avg.get('body_line', 0),
                        'frames':             n,
                    })
                    print(f"[REP {rep_num}] {best_label} {best_conf*100:.0f}% ({n} frames)")

                label = current_form.get('label', 'N/A')
                conf2 = current_form.get('confidence', 0.0)

                if is_in_pos:
                    state_text = si['state']
                    if label != 'N/A' and rep_num > 0:
                        color_bgr = classifier.get_color(label)
                        ov = annotated.copy()
                        cv2.rectangle(ov, (0, 0), (w, 50), color_bgr, -1)
                        annotated = cv2.addWeighted(annotated, 0.6, ov, 0.4, 0)
                        cv2.putText(annotated,
                                    f"Last: {label.upper().replace('_',' ')}  {conf2*100:.0f}%",
                                    (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    else:
                        ov = annotated.copy()
                        cv2.rectangle(ov, (0, 0), (w, 50), (0, 100, 80), -1)
                        annotated = cv2.addWeighted(annotated, 0.6, ov, 0.4, 0)
                        cv2.putText(annotated, f"IN POSITION - {state_text}",
                                    (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                else:
                    ov = annotated.copy()
                    cv2.rectangle(ov, (0, 0), (w, 50), (60, 60, 60), -1)
                    annotated = cv2.addWeighted(annotated, 0.6, ov, 0.4, 0)
                    cv2.putText(annotated, "Get into pushup position",
                                (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

                ov2 = annotated.copy()
                cv2.rectangle(ov2, (0, h-50), (w, h), (30,30,30), -1)
                annotated = cv2.addWeighted(annotated, 0.65, ov2, 0.35, 0)
                feedback_text = classifier.get_feedback(label) if (is_in_pos and rep_num > 0) else ''
                cv2.putText(annotated, feedback_text,
                            (15, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(annotated, f"Reps: {rep_num}",
                            (w-110, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.circle(annotated, (w-15, 15), 7, (0,220,0), -1)

            else:
                current_position_status = {
                    'in_position': False, 'confidence': 0.0, 'message': 'No pose'}
                annotated = frame

            last_annotated = annotated
            ret2, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret2:
                with frame_lock:
                    output_frame = buf.tobytes()

        except Exception as e:
            print(f"[process_loop error] {e}")
            time.sleep(0.01)


def generate_frames():
    interval = 1.0 / 20
    while True:
        t0 = time.time()
        with frame_lock:
            frame = output_frame
        if frame is not None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        wait = interval - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_angles')
def get_angles():
    return jsonify(current_angles)

@app.route('/get_position_status')
def get_position_status():
    return jsonify(current_position_status)

@app.route('/get_state_info')
def get_state_info():
    return jsonify(current_state_info)

@app.route('/get_form')
def get_form():
    return jsonify(current_form)

@app.route('/get_rep_history')
def get_rep_history():
    return jsonify(rep_history)

@app.route('/get_beep')
def get_beep():
    global beep_pending
    b = beep_pending
    beep_pending = False
    return jsonify({'beep': b})

@app.route('/get_calibration_status')
def get_calibration_status():
    return jsonify(calibration_status)

@app.route('/reset_reps', methods=['POST'])
def reset_reps():
    global rep_history, last_rep_seen, beep_pending
    state_machine.reset()
    classifier.reset()
    rep_history.clear()
    last_rep_seen   = 0
    beep_pending    = False
    return jsonify({'success': True})

@app.route('/recalibrate', methods=['POST'])
def recalibrate():
    global calib_active, calib_last_rep, calib_message, BASELINE
    global calib_rep_data, calib_sm

    for f in [CALIB_JSON, CALIB_PROFILE]:
        if os.path.exists(f):
            os.remove(f)

    calib_active = True
    calib_sm     = PushupStateMachine()
    calib_rep_data.clear()
    calib_last_rep = 0
    calib_elbow_hist.clear()
    calib_hip_hist.clear()
    user_calib.reset()
    calib_message = "Do 5 slow perfect pushups to calibrate"
    calibration_status['done'] = False
    calibration_status['reps_done'] = 0
    calibration_status['message'] = 'Calibration needed'

    BASELINE = load_baseline()
    return jsonify({'success': True})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'calibrated': os.path.exists(CALIB_JSON)})


if __name__ == '__main__':
    t_cam = threading.Thread(target=camera_reader, daemon=True)
    t_cam.start()
    time.sleep(0.5)

    t_proc = threading.Thread(target=process_loop, daemon=True)
    t_proc.start()
    time.sleep(0.5)

    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)