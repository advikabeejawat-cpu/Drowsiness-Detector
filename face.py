# ==========================================
# MAIN PROGRAM
# ==========================================
#
# Focus Buddy
#
# Webcam
#    ↓
# MediaPipe
#    ↓
# Detection
#    ↓
# Calibration
#    ↓
# State Machine
#    ↓
# Event Logger
#    ↓
# Session Summary
# ==========================================


import cv2
import mediapipe as mp
import time
import winsound
import threading


# ==========================================
# PROJECT MODULES
# ==========================================

from detection import (
    get_eye_status,
    get_head_position
)

from calibration import Calibration

from state_machine import DetectionState

from ui import (
    draw_ui,
    draw_calibration_ui
)

from event_logger import EventLogger
from behavior_predictor import BehaviorPredictor


# ==========================================
# SETTINGS
# ==========================================

from settings import (
    EYE_CLOSED_TIME,
    HEAD_TURN_TIME,
    WARNING_INTERVAL,
    BEEP_FREQUENCY,
    BEEP_DURATION
)


MAX_FACES = 5


def _face_center_x(face_landmarks):
    xs = [landmark.x for landmark in face_landmarks.landmark]
    return (min(xs) + max(xs)) / 2.0


def _draw_face_overlays(image, overlays):
    height, width = image.shape[:2]
    for person_number, face_landmarks, label in overlays:
        xs = [landmark.x for landmark in face_landmarks.landmark]
        ys = [landmark.y for landmark in face_landmarks.landmark]
        x1 = max(0, int(min(xs) * width) - 8)
        y1 = max(0, int(min(ys) * height) - 8)
        x2 = min(width - 1, int(max(xs) * width) + 8)
        y2 = min(height - 1, int(max(ys) * height) + 8)
        color = (70, 210, 100) if "Focused" in label else (60, 170, 240)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"Person {person_number}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )


def _log_person_events(logger, events, session_start_time, person_number):
    for event in events:
        start_time = event["start"] - session_start_time
        end_time = event["end"] - session_start_time
        logger.log_event(event["type"], start_time, end_time, person_number)
        print(
            f"Person {person_number} EVENT: {event['type']} "
            f"({end_time - start_time:.2f}s)"
        )


def play_warning_beep():
    """Play a non-blocking Windows notification sound."""
    try:
        winsound.PlaySound(
            "SystemExclamation",
            winsound.SND_ALIAS | winsound.SND_ASYNC
        )
    except RuntimeError:
        # Keep a frequency-based fallback for systems without the alias.
        winsound.Beep(BEEP_FREQUENCY, BEEP_DURATION)


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Could not open webcam.")

    exit()


# ==========================================
# MEDIAPIPE
# ==========================================

mp_face_mesh = mp.solutions.face_mesh


face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=MAX_FACES,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# LOGGER
# ==========================================

logger = EventLogger()
predictor = BehaviorPredictor()


# ==========================================
# CALIBRATION
# ==========================================

calibration = Calibration()

calibration.start()


# ==========================================
# STATE MACHINE
# ==========================================

detector = DetectionState()
additional_detectors = {}
additional_head_positions = {}
additional_tracking_lost_starts = {}
additional_calibrations = {}


# ==========================================
# HEAD SMOOTHING
# ==========================================

head_positions = []


# ==========================================
# TRACKING
# ==========================================

tracking_lost_start_time = None


# ==========================================
# WARNING TIMER
# ==========================================

last_warning_time = 0


# ==========================================
# SESSION TIMER
# ==========================================

session_start_time = time.time()


# ==========================================
# MAIN LOOP
# ==========================================

while cap.isOpened():

    # --------------------------------------
    # Read frame
    # --------------------------------------

    success, image = cap.read()


    if not success:

        print("Could not read from webcam.")

        break


    current_time = time.time()


    # --------------------------------------
    # Image dimensions
    # --------------------------------------

    image_height, image_width, _ = (
        image.shape
    )


    # --------------------------------------
    # MediaPipe
    # --------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    results = face_mesh.process(
        rgb_image
    )

    detected_faces = sorted(
        results.multi_face_landmarks or [],
        key=_face_center_x,
    )
    face_overlays = []
    people_statuses = []
    active_additional_ids = set()


    # ======================================
    # CALIBRATION
    # ======================================

    if calibration.active:

        face_detected = bool(detected_faces)


        if face_detected:

            face_landmarks = (
                detected_faces[0]
            )


            # Get normal eye measurement.
            (
                eye_status,
                average_ear,
                _,
                _
            ) = get_eye_status(
                face_landmarks,
                image_width,
                image_height
            )


            # Get normal head measurement.
            (
                head_status,
                head_position,
                head_positions
            ) = get_head_position(
                face_landmarks,
                head_positions
            )


            # Give measurements to calibration.
            calibration.update(
                average_ear,
                head_position
            )

            for person_number, person_landmarks in enumerate(detected_faces[1:], start=2):
                person_calibration = additional_calibrations.setdefault(
                    person_number, Calibration()
                )
                if not person_calibration.active and not person_calibration.complete:
                    person_calibration.start()
                person_ear_status, person_ear, _, _ = get_eye_status(
                    person_landmarks, image_width, image_height
                )
                person_positions = additional_head_positions.setdefault(person_number, [])
                _, person_head_position, person_positions = get_head_position(
                    person_landmarks, person_positions
                )
                additional_head_positions[person_number] = person_positions
                person_calibration.update(person_ear, person_head_position)

        face_overlays = [
            (person_number, landmarks, "Calibrating")
            for person_number, landmarks in enumerate(detected_faces, start=1)
        ]


        # Draw calibration screen.
        image = draw_calibration_ui(
            image,
            calibration.get_progress(),
            calibration.get_remaining_time(),
            face_detected,
            ear_sample_count=len(calibration.ear_values),
            ear_baseline=calibration.get_eye_baseline(),
            threshold_preview=calibration.get_eye_threshold_preview(),
        )


    # ======================================
    # NORMAL MONITORING
    # ======================================

    else:

        eye_status = "Unknown"

        average_ear = None

        eye_threshold = calibration.get_eye_threshold()

        head_status = "Unknown"

        closed_duration = 0.0

        head_turn_duration = 0.0

        drowsy_warning = False

        attention_warning = False

        main_status = "No Face"
        prediction_text = "Learning your patterns from past sessions..."
        prediction_detail = ""


        # ==================================
        # FACE DETECTED
        # ==================================

        if detected_faces:

            face_landmarks = (
                detected_faces[0]
            )


            # ----------------------------------
            # TRACKING RETURNED
            # ----------------------------------

            if tracking_lost_start_time is not None:

                logger.log_event(
                    "Tracking Lost",

                    tracking_lost_start_time
                    - session_start_time,

                    current_time
                    - session_start_time,
                    person_number=1,
                )

                tracking_lost_start_time = None


            # ==================================
            # PERSONALIZED THRESHOLDS
            # ==================================

            eye_threshold = (
                calibration.get_eye_threshold()
            )

            head_thresholds = (
                calibration.get_head_thresholds()
            )


            # ==================================
            # EYE DETECTION
            # ==================================

            (
                eye_status,
                average_ear,
                _,
                _
            ) = get_eye_status(
                face_landmarks,
                image_width,
                image_height,
                eye_threshold
            )


            # ==================================
            # HEAD DETECTION
            # ==================================

            (
                head_status,
                head_position,
                head_positions
            ) = get_head_position(
                face_landmarks,
                head_positions,
                head_thresholds
            )


            # ==================================
            # STATE MACHINE
            # ==================================

            state = detector.update(
                eye_status,
                head_status
            )
            people_statuses.append({
                "number": 1,
                "main_state": state["main_state"],
                "eye_status": eye_status,
                "head_status": head_status,
                "closed_duration": state["closed_duration"],
                "head_turn_duration": state["head_turn_duration"],
                "average_ear": average_ear,
                "eye_threshold": eye_threshold,
            })
            face_overlays.append((
                1,
                face_landmarks,
                f"{state['main_state']} | eyes {eye_status.lower()} | {head_status.lower()}",
            ))

            recent_behavior = list(logger.events)
            if detector.head_away_start is not None:
                away_elapsed = current_time - session_start_time
                recent_behavior.append({
                    "type": "Looking Away",
                    "start": detector.head_away_start - session_start_time,
                    "end": away_elapsed,
                })
            prediction = predictor.predict(
                current_time - session_start_time,
                recent_behavior
            )
            if prediction["ready"]:
                chance = prediction["probability"]
                if chance >= 0.60:
                    prediction_text = f"Elevated distraction risk in next 10s ({chance:.0%})"
                    if prediction["driver"] == "recent looking-away events":
                        prediction_detail = "Recent look-away pattern: recenter and refocus for a moment."
                    elif prediction["driver"] == "recent prolonged eye-closure events":
                        prediction_detail = "Recent prolonged eye closures: take a short break and refocus."
                    elif prediction["driver"] == "recent tracking interruptions":
                        prediction_detail = "Tracking pattern: check camera framing and lighting."
                    else:
                        prediction_detail = "This early estimate is based on your saved sessions."
                else:
                    prediction_text = f"Low distraction risk in next 10s ({chance:.0%})"
                    prediction_detail = "Compared with your past sessions; blinks do not lower this score."
            else:
                prediction_text = (
                    "Building your personal forecast "
                    f"({prediction['samples']} history windows)"
                )


            main_status = (
                state["main_state"]
            )


            closed_duration = (
                state["closed_duration"]
            )


            head_turn_duration = (
                state["head_turn_duration"]
            )


            drowsy_warning = (
                state["drowsy_warning"]
            )


            attention_warning = (
                state["attention_warning"]
            )


            # ==================================
            # SAVE COMPLETED EVENTS
            # ==================================

            for event in state["events"]:

                start_time = (
                    event["start"]
                    - session_start_time
                )

                end_time = (
                    event["end"]
                    - session_start_time
                )


                logger.log_event(
                    event["type"],
                    start_time,
                    end_time,
                    person_number=1,
                )


                print(
                    f"EVENT: {event['type']} "
                    f"({end_time - start_time:.2f}s)"
                )


            # ==================================
            # WARNING BEEP
            # ==================================

            if (
                drowsy_warning
                or attention_warning
            ):

                if (
                    current_time
                    - last_warning_time
                    >= WARNING_INTERVAL
                ):

                    last_warning_time = (
                        current_time
                    )
                    threading.Thread(
                        target=play_warning_beep,
                        daemon=True
                    ).start()
                    print(
                        "ALERT: "
                        + (
                            "eyes closed for 2+ seconds"
                            if drowsy_warning
                            else "looking away for 2+ seconds"
                        )
                    )
            else:
                last_warning_time = 0

            # Keep a separate state machine and head smoother for every
            # additional face. IDs follow left-to-right position in the frame.
            for person_number, person_landmarks in enumerate(detected_faces[1:], start=2):
                active_additional_ids.add(person_number)
                person_detector = additional_detectors.setdefault(
                    person_number, DetectionState()
                )
                person_positions = additional_head_positions.setdefault(person_number, [])
                person_calibration = additional_calibrations.setdefault(
                    person_number, Calibration()
                )
                if not person_calibration.active and not person_calibration.complete:
                    person_calibration.start()

                if person_number in additional_tracking_lost_starts:
                    lost_start = additional_tracking_lost_starts.pop(person_number)
                    logger.log_event(
                        "Tracking Lost",
                        lost_start - session_start_time,
                        current_time - session_start_time,
                        person_number=person_number,
                    )

                person_eye_threshold = (
                    person_calibration.get_eye_threshold()
                    if person_calibration.complete
                    else eye_threshold
                )
                person_head_thresholds = (
                    person_calibration.get_head_thresholds()
                    if person_calibration.complete
                    else head_thresholds
                )
                person_eye_status, person_ear, _, _ = get_eye_status(
                    person_landmarks,
                    image_width,
                    image_height,
                    person_eye_threshold,
                )
                person_head_status, person_head_position, person_positions = get_head_position(
                    person_landmarks,
                    person_positions,
                    person_head_thresholds,
                )
                additional_head_positions[person_number] = person_positions
                if person_calibration.active:
                    person_calibration.update(person_ear, person_head_position)
                person_state = person_detector.update(
                    person_eye_status,
                    person_head_status,
                )
                people_statuses.append({
                    "number": person_number,
                    "main_state": person_state["main_state"],
                    "eye_status": person_eye_status,
                    "head_status": person_head_status,
                    "closed_duration": person_state["closed_duration"],
                    "head_turn_duration": person_state["head_turn_duration"],
                    "average_ear": person_ear,
                    "eye_threshold": person_eye_threshold,
                })
                _log_person_events(
                    logger, person_state["events"], session_start_time, person_number
                )
                face_overlays.append((
                    person_number,
                    person_landmarks,
                    f"{person_state['main_state']} | eyes {person_eye_status.lower()} | "
                    f"{person_head_status.lower()}",
                ))
                if (
                    person_state["drowsy_warning"]
                    or person_state["attention_warning"]
                ) and current_time - last_warning_time >= WARNING_INTERVAL:
                    last_warning_time = current_time
                    threading.Thread(target=play_warning_beep, daemon=True).start()
                    print(f"ALERT: Person {person_number} needs refocusing")


        # ==================================
        # NO FACE
        # ==================================

        else:

            main_status = "No Face"

            last_warning_time = 0

            eye_status = "Unknown"

            head_status = "Unknown"
            prediction_text = "Prediction paused while face tracking is lost"
            prediction_detail = ""


            # Start tracking-loss timer.
            if tracking_lost_start_time is None:

                tracking_lost_start_time = (
                    current_time
                )


            # We cannot safely determine head
            # position anymore.
            head_positions.clear()


            # Reset state machine.
            detector.reset()

        # If an additional face disappears, finish any active event and start
        # a tracking-loss interval for that positional ID.
        for person_number, person_detector in additional_detectors.items():
            if person_number in active_additional_ids:
                continue
            if person_number not in additional_tracking_lost_starts:
                additional_tracking_lost_starts[person_number] = current_time
            missing_state = person_detector.update("No Face", "No Face")
            _log_person_events(
                logger, missing_state["events"], session_start_time, person_number
            )
            additional_head_positions.get(person_number, []).clear()


        # ==================================
        # DRAW UI
        # ==================================

        image = draw_ui(
            image,
            main_status,
            calibrated=calibration.complete,
            prediction_text=prediction_text,
            prediction_detail=prediction_detail,
            people_statuses=people_statuses,
        )

    _draw_face_overlays(image, face_overlays)


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "focus buddy",
        image
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# SESSION END
# ==========================================

session_end_time = time.time()


session_duration = (
    session_end_time
    - session_start_time
)


# ==========================================
# FINISH ACTIVE EYE EVENT
# ==========================================

if detector.eye_closed_start is not None:

    duration = (
        session_end_time
        - detector.eye_closed_start
    )


    if duration >= EYE_CLOSED_TIME:

        logger.log_event(
            "Prolonged Eye Closure",

            detector.eye_closed_start
            - session_start_time,

            session_duration,
            person_number=1,
        )

    else:

        logger.log_event(
            "Blink",

            detector.eye_closed_start
            - session_start_time,

            session_duration,
            person_number=1,
        )


# ==========================================
# FINISH ACTIVE HEAD EVENT
# ==========================================

if detector.head_away_start is not None:

    duration = (
        session_end_time
        - detector.head_away_start
    )


    if duration >= HEAD_TURN_TIME:

        logger.log_event(
            "Looking Away",

            detector.head_away_start
            - session_start_time,

            session_duration,
            person_number=1,
        )


# ==========================================
# FINISH ADDITIONAL PEOPLE'S EVENTS
# ==========================================

for person_number, person_detector in additional_detectors.items():
    final_state = person_detector.update("No Face", "No Face")
    _log_person_events(
        logger,
        final_state["events"],
        session_start_time,
        person_number,
    )
    lost_start = additional_tracking_lost_starts.get(person_number)
    if lost_start is not None:
        logger.log_event(
            "Tracking Lost",
            lost_start - session_start_time,
            session_duration,
            person_number=person_number,
        )


# ==========================================
# FINISH TRACKING EVENT
# ==========================================

if tracking_lost_start_time is not None:

    logger.log_event(
        "Tracking Lost",

        tracking_lost_start_time
        - session_start_time,

        session_duration,
        person_number=1,
    )


# ==========================================
# SAVE CSV
# ==========================================

personal_insight = predictor.session_insight(logger.events)
print("PERSONAL FOCUS INSIGHT: " + personal_insight)

filename = logger.save_csv(session_duration)
predictor.retrain()
if predictor.model is None:
    print(
        "Behavior predictor is still learning: "
        f"{predictor.sample_count} usable time windows, "
        f"{predictor.positive_count} distraction windows."
    )
else:
    print(
        "Behavior predictor trained from "
        f"{predictor.sample_count} saved time windows "
        f"({predictor.positive_count} labeled as distraction-risk windows)."
    )


# SESSION SUMMARY
# ==========================================

logger.print_summary(session_duration)

print("\n")
print("=" * 50)
print("          PERSONAL CALIBRATION")
print("=" * 50)

summary = calibration.get_summary()
if summary["baseline_ear"] is not None:
    print(f"\nBaseline EAR: {summary['baseline_ear']:.3f}")
    print(f"Personal eye threshold: {summary['eye_threshold']:.3f}")
else:
    print("\nNo eye calibration data collected.")

if summary["baseline_head"] is not None:
    print(f"\nBaseline head position: {summary['baseline_head']:.3f}")
    print(f"Personal head range: {summary['head_low']:.3f} - {summary['head_high']:.3f}")
else:
    print("\nNo head calibration data collected.")

print(f"\nCalibration samples: {summary['samples']}")
print("=" * 50)
print(f"\nSession data saved to: {filename}")

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

