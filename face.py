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
    max_num_faces=1,
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


    # ======================================
    # CALIBRATION
    # ======================================

    if calibration.active:

        face_detected = bool(
            results.multi_face_landmarks
        )


        if face_detected:

            face_landmarks = (
                results.multi_face_landmarks[0]
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


        # Draw calibration screen.
        image = draw_calibration_ui(
            image,
            calibration.get_progress(),
            calibration.get_remaining_time(),
            face_detected
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

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
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
                    - session_start_time
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
                    end_time
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


        # ==================================
        # DRAW UI
        # ==================================

        image = draw_ui(
            image,
            main_status,
            eye_status,
            head_status,
            closed_duration,
            head_turn_duration,
            drowsy_warning,
            attention_warning,
            calibrated=calibration.complete,
            prediction_text=prediction_text,
            prediction_detail=prediction_detail,
            average_ear=average_ear,
            eye_threshold=eye_threshold
        )


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

            session_duration
        )

    else:

        logger.log_event(
            "Blink",

            detector.eye_closed_start
            - session_start_time,

            session_duration
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

            session_duration
        )


# ==========================================
# FINISH TRACKING EVENT
# ==========================================

if tracking_lost_start_time is not None:

    logger.log_event(
        "Tracking Lost",

        tracking_lost_start_time
        - session_start_time,

        session_duration
    )


# ==========================================
# SAVE CSV
# ==========================================

personal_insight = predictor.session_insight(logger.events)
print("PERSONAL FOCUS INSIGHT: " + personal_insight)

filename = logger.save_csv()
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
        f"{predictor.sample_count} historical time windows "
        f"({predictor.positive_count} followed by a look-away event)."
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

