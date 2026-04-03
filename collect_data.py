import cv2
import csv
import json
import time
import numpy as np
import os
from collections import deque, defaultdict
from pose_analyzer import PoseAnalyzer
from exercise_detector import ExerciseDetector
from state_machine import PushupStateMachine
from user_calibration import UserCalibration

OUTPUT_CSV       = 'pushup_data_v2.csv'
CALIBRATION_FILE = 'calibration_profile.json'
TOLERANCE        = 10.0
HIP_TOLERANCE    = 15.0
CALIB_REPS       = 5

# Global user calibration instance
user_calibration = UserCalibration()

FIELDNAMES = [
    'elbow', 'elbow_flare', 'hip', 'wrist', 'shoulder_elevation', 'body_line',
    'elbow_velocity', 'hip_velocity',
    'time_in_phase', 'rep_number',
    'elbow_drift', 'hip_drift', 'velocity_trend',
    'rolling_elbow_5', 'rolling_hip_5',
    'phase', 'form_quality', 'subject_id'
]

STATE_PHASE = {
    'TOP': 'up', 'BOTTOM': 'down',
    'DESCENDING': 'mid', 'ASCENDING': 'mid',
}

FORM_KEYS = {
    ord('p'): 'perfect',
    ord('h'): 'hip_sag',
    ord('f'): 'fatigue_hip_sag',
    ord('e'): 'elbow_flare',
    ord('s'): 'shoulder_dip',
    ord('x'): 'shallow',
}

def compute_velocity(history):
    if len(history) < 2:
        return 0.0
    hist = list(history)
    return float(np.mean([(hist[i] - hist[i-1]) * 30 for i in range(1, len(hist))]))

def load_existing_counts(csv_path):
    counts = defaultdict(int)
    if not os.path.exists(csv_path):
        return counts
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            counts[row['form_quality']] += 1
    return counts

def get_subject_id():
    print("\n=== SUBJECT SETUP ===")
    while True:
        try:
            sid = int(input("Enter subject ID (e.g. 5, 6, 7 — must be unique per person): "))
            return sid
        except ValueError:
            print("Please enter a number.")

def draw_text(img, text, pos, scale=0.65, color=(255,255,255), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def overlay_box(img, y1, y2, alpha=0.6):
    h, w = img.shape[:2]
    dark = img.copy()
    cv2.rectangle(dark, (0, y1), (w, y2), (0,0,0), -1)
    return cv2.addWeighted(img, 1-alpha, dark, alpha, 0)

def run_calibration(cap, analyzer, detector, sm):
    print(f"\n=== CALIBRATION PHASE ===")
    print(f"Do {CALIB_REPS} slow perfect reps. Rep 1 will be discarded.\n")

    rep_data = defaultdict(lambda: defaultdict(list))
    last_rep = 0
    elbow_hist = deque(maxlen=10)
    hip_hist   = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        annotated, angles, landmarks = analyzer.process_frame(frame)

        if angles:
            is_in_pos, _, _ = detector.is_pushup_position(angles, landmarks)
            info  = sm.update(angles, is_in_pos)
            state = info['state']
            rep   = info['rep_count']

            elbow_hist.append(angles['elbow'])
            hip_hist.append(angles['hip'])

            # Feed into user calibration buffer (perfect reps only)
            if rep >= 2:   # skip rep 1 same as baseline
                user_calibration.add_calibration_sample(angles)

            phase = STATE_PHASE.get(state)
            if phase and rep > last_rep:
                last_rep = rep

            if phase and rep > 0:
                rep_data[rep][phase + '_elbow'].append(angles['elbow'])
                rep_data[rep][phase + '_hip'].append(angles['hip'])
                rep_data[rep][phase + '_vel'].append(compute_velocity(elbow_hist))
                rep_data[rep][phase + '_body'].append(angles['body_line'])

            collected = len(rep_data)
            annotated = overlay_box(annotated, 0, 160)
            draw_text(annotated, "CALIBRATION PHASE", (10, 30), color=(0,255,255))
            draw_text(annotated, f"Do {CALIB_REPS} slow perfect reps (rep 1 discarded)", (10, 60), scale=0.55)
            draw_text(annotated, f"Reps collected: {collected}/{CALIB_REPS}", (10, 90), scale=0.55)
            draw_text(annotated, f"State: {state}  Phase: {phase or 'skip'}", (10, 120))
            if collected >= CALIB_REPS:
                draw_text(annotated, "Done! Press ENTER to continue or R to redo", (10, 150), color=(0,255,0), scale=0.55)

        cv2.imshow('RehabMate - Data Collection', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        if len(rep_data) >= CALIB_REPS:
            if key == 13:
                break
            if key == ord('r'):
                rep_data.clear()
                last_rep = 0
                sm.reset()

    valid_reps = [r for r in rep_data if r >= 2]
    if not valid_reps:
        return None

    def avg(key):
        vals = []
        for r in valid_reps:
            vals.extend(rep_data[r].get(key, []))
        return float(np.mean(vals)) if vals else None

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

    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"\nCalibration saved → {CALIBRATION_FILE}")
    print(f"  Elbow top: {baseline['elbow_top']:.1f}°  bottom: {baseline['elbow_bottom']:.1f}°")
    print(f"  Hip:       {baseline['hip_top']:.1f}°    tolerance: ±{TOLERANCE}° elbow / ±{HIP_TOLERANCE}° hip")

    # Finalize user-level feature calibration
    user_calibration.finalize_calibration(save_path='user_calibration.json')

    return baseline

def confirm_baseline(cap, analyzer, baseline):
    print("\nConfirm baseline. Press ENTER to accept or R to redo.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        annotated, _, _ = analyzer.process_frame(frame)

        annotated = overlay_box(annotated, 0, 220)
        draw_text(annotated, f"YOUR BASELINE (±{TOLERANCE}° tolerance)", (10, 30), color=(0,255,255))
        draw_text(annotated, f"Elbow top:    {baseline['elbow_top']:.1f}°", (10, 65), scale=0.6)
        draw_text(annotated, f"Elbow bottom: {baseline['elbow_bottom']:.1f}°", (10, 95), scale=0.6)
        draw_text(annotated, f"Hip:          {baseline['hip_top']:.1f}°", (10, 125), scale=0.6)
        draw_text(annotated, f"Body line:    {baseline['body_line']:.1f}°", (10, 155), scale=0.6)
        draw_text(annotated, "ENTER = accept    R = redo", (10, 195), color=(0,255,0), scale=0.6)

        cv2.imshow('RehabMate - Data Collection', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            return True
        if key == ord('r'):
            return False
        if key == ord('q'):
            return None

def detect_form_quality(angles, phase, elbow_vel, baseline, rep_number, rolling_elbow, rolling_hip):
    tol     = baseline['tolerance']
    hip_tol = baseline['hip_tolerance']

    if phase == 'down':
        elbow_ref = baseline['elbow_bottom']
        hip_ref   = baseline['hip_bottom']
    elif phase == 'up':
        elbow_ref = baseline['elbow_top']
        hip_ref   = baseline['hip_top']
    else:
        elbow_ref = baseline['elbow_mid']
        hip_ref   = baseline['hip_mid']

    elbow_drift = angles['elbow'] - elbow_ref
    hip_drift   = angles['hip']   - hip_ref
    vel_drift   = elbow_vel - baseline['velocity_desc'] if baseline['velocity_desc'] else 0

    if rep_number >= 5 and rolling_elbow is not None:
        elbow_drift_roll = rolling_elbow - elbow_ref
        hip_drift_roll   = rolling_hip   - hip_ref
        if phase == 'down' and elbow_drift_roll > tol:
            return 'shallow'
        if hip_drift_roll < -hip_tol:
            return 'fatigue_hip_sag'

    if phase == 'down' and elbow_drift > tol:
        return 'shallow'
    if angles['hip'] < hip_ref - hip_tol:
        return 'hip_sag'
    if angles['elbow_flare'] > 55:
        return 'elbow_flare'
    if angles['shoulder_elevation'] > 6:
        return 'shoulder_dip'

    return 'perfect'

def run_collection(cap, analyzer, detector, sm, baseline, form_counts, writer, f, subject_id):
    print("\n=== COLLECTION PHASE ===")
    print("Phase 2: Do pushups till failure (auto-labeled)")
    print("Phase 3: After failure, do intentional bad form")
    print("  P=perfect  H=hip_sag  F=fatigue_hip_sag  E=elbow_flare  S=shoulder_dip  X=shallow  Space=skip  Q=quit\n")

    elbow_hist   = deque(maxlen=10)
    hip_hist     = deque(maxlen=10)
    rep_elbow    = deque(maxlen=5)
    rep_hip      = deque(maxlen=5)
    rep_vel_hist = deque(maxlen=10)

    rep_buffer  = []
    last_rep    = 0
    waiting     = False
    phase2_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        annotated, angles, landmarks = analyzer.process_frame(frame)

        if angles:
            is_in_pos, _, _ = detector.is_pushup_position(angles, landmarks)
            info  = sm.update(angles, is_in_pos)
            state = info['state']
            rep   = info['rep_count']
            phase = STATE_PHASE.get(state)

            elbow_hist.append(angles['elbow'])
            hip_hist.append(angles['hip'])
            elbow_vel = compute_velocity(elbow_hist)
            hip_vel   = compute_velocity(hip_hist)
            rep_vel_hist.append(elbow_vel)

            roll_elbow = float(np.mean(rep_elbow)) if rep_elbow else None
            roll_hip   = float(np.mean(rep_hip))   if rep_hip   else None
            vel_trend  = float(np.mean(rep_vel_hist)) if rep_vel_hist else 0

            elbow_drift = angles['elbow'] - (baseline['elbow_bottom'] if phase == 'down' else baseline['elbow_top'])
            hip_drift   = angles['hip']   - (baseline['hip_bottom']   if phase == 'down' else baseline['hip_top'])

            if rep > last_rep:
                last_rep = rep
                bot_frames = [r for r in rep_buffer if r['phase'] == 'down']
                if bot_frames:
                    rep_elbow.append(np.mean([r['elbow'] for r in bot_frames]))
                    rep_hip.append(np.mean([r['hip']    for r in bot_frames]))
                waiting = True

            if phase and not waiting:
                rep_buffer.append({
                    'elbow':              round(angles['elbow'], 2),
                    'elbow_flare':        round(angles['elbow_flare'], 2),
                    'hip':                round(angles['hip'], 2),
                    'wrist':              180.0,
                    'shoulder_elevation': round(angles['shoulder_elevation'], 2),
                    'body_line':          round(angles['body_line'], 2),
                    'elbow_velocity':     round(elbow_vel, 2),
                    'hip_velocity':       round(hip_vel, 2),
                    'time_in_phase':      round(info['state_duration'], 3),
                    'rep_number':         rep,
                    'elbow_drift':        round(elbow_drift, 2),
                    'hip_drift':          round(hip_drift, 2),
                    'velocity_trend':     round(vel_trend, 2),
                    'rolling_elbow_5':    round(roll_elbow, 2) if roll_elbow else 0.0,
                    'rolling_hip_5':      round(roll_hip, 2)   if roll_hip   else 0.0,
                    'phase':              phase,
                    'form_quality':       None,
                    'subject_id':         subject_id,
                })

            annotated = overlay_box(annotated, 0, 50)
            draw_text(annotated, f"Subject:{subject_id}  State:{state}  Rep:{rep}  Phase:{phase or 'skip'}", (10, 30))

            annotated = overlay_box(annotated, 380, 480, alpha=0.5)
            draw_text(annotated, f"perfect:{form_counts['perfect']}  hip_sag:{form_counts['hip_sag']}  fatigue:{form_counts['fatigue_hip_sag']}", (10, 400), scale=0.5)
            draw_text(annotated, f"elbow_flare:{form_counts['elbow_flare']}  shoulder_dip:{form_counts['shoulder_dip']}  shallow:{form_counts['shallow']}", (10, 425), scale=0.5)
            draw_text(annotated, f"Elbow drift:{elbow_drift:+.1f}°  Hip drift:{hip_drift:+.1f}°", (10, 450), scale=0.5, color=(255,255,0))
            draw_text(annotated, f"Buffer: {len(rep_buffer)} frames", (10, 475), scale=0.5)

            if waiting:
                annotated = overlay_box(annotated, 60, 300)
                draw_text(annotated, f"Rep {rep} done! Label:", (10, 90), color=(0,255,255))
                if not phase2_done:
                    draw_text(annotated, "ENTER = auto-label", (10, 125), color=(0,255,0), scale=0.6)
                    draw_text(annotated, "F = switch to manual mode", (10, 155), color=(255,165,0), scale=0.6)
                    draw_text(annotated, "Space = skip", (10, 185), scale=0.6)
                else:
                    draw_text(annotated, "P=perfect  H=hip_sag  F=fatigue", (10, 125), color=(0,255,255), scale=0.6)
                    draw_text(annotated, "E=elbow_flare  S=shoulder_dip  X=shallow", (10, 155), color=(0,255,255), scale=0.6)
                    draw_text(annotated, "Space=skip", (10, 185), scale=0.55)

        cv2.imshow('RehabMate - Data Collection', annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if waiting and angles:
            if not phase2_done and key == 13:
                form_label = detect_form_quality(
                    angles, phase, elbow_vel, baseline,
                    last_rep, roll_elbow, roll_hip
                )
                for row in rep_buffer:
                    row['form_quality'] = form_label
                    writer.writerow(row)
                f.flush()
                form_counts[form_label] += len(rep_buffer)
                print(f"Rep {last_rep}: auto → {form_label} ({len(rep_buffer)} frames)")
                rep_buffer.clear()
                waiting = False

            elif not phase2_done and key == ord('m'):
                phase2_done = True
                rep_buffer.clear()
                waiting = False
                print("Switched to manual labeling")

            elif phase2_done:
                form_label = FORM_KEYS.get(key)
                if form_label:
                    for row in rep_buffer:
                        row['form_quality'] = form_label
                        writer.writerow(row)
                    f.flush()
                    form_counts[form_label] += len(rep_buffer)
                    print(f"Rep {last_rep}: manual → {form_label} ({len(rep_buffer)} frames)")
                    rep_buffer.clear()
                    waiting = False

            if key == ord(' '):
                print(f"Rep {last_rep}: skipped")
                rep_buffer.clear()
                waiting = False

def main():
    subject_id   = get_subject_id()
    analyzer     = PoseAnalyzer()
    detector     = ExerciseDetector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    form_counts  = load_existing_counts(OUTPUT_CSV)
    write_header = not os.path.exists(OUTPUT_CSV)

    baseline = None
    while baseline is None:
        sm       = PushupStateMachine()
        baseline = run_calibration(cap, analyzer, detector, sm)
        if baseline is None:
            break
        result = confirm_baseline(cap, analyzer, baseline)
        if result is None:
            break
        if result is False:
            baseline = None

    if baseline is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    sm = PushupStateMachine()

    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        run_collection(cap, analyzer, detector, sm, baseline, form_counts, writer, f, subject_id)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal counts: {dict(form_counts)}")

if __name__ == '__main__':
    main()