
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Face coordinates for left eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Face coordinates for right eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Face coordinates for head direction
NOSE = 1
LEFT_FACE = 234
RIGHT_FACE = 454

EAR_THRESHOLD = 0.20
EYE_CLOSED_TIME = 2.0

# How long the user can look away before the warning starts
HEAD_TURN_TIME = 2.0

# How often the warning beep repeats
WARNING_INTERVAL = 3.0

# Threshold for detecting when the head is turned
HEAD_TURN_THRESHOLD = 0.35

closed_start_time = None
head_turn_start_time = None

last_warning_time = None
last_head_warning_time = None


def calculate_ear(eye_points):
    vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():

        success, image = cap.read()

        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:

            custom_spec = mp_drawing.DrawingSpec(
                color=(230, 216, 173),
                thickness=1,
                circle_radius=1
            )

            for face_landmarks in results.multi_face_landmarks:

                # --------------------------------
                # HEAD DIRECTION
                # --------------------------------

                nose = face_landmarks.landmark[NOSE]
                left_face = face_landmarks.landmark[LEFT_FACE]
                right_face = face_landmarks.landmark[RIGHT_FACE]

                face_width = right_face.x - left_face.x

                nose_position = (nose.x - left_face.x) / face_width

                if nose_position < HEAD_TURN_THRESHOLD:
                    head_status = "Looking Away"

                    if head_turn_start_time is None:
                        head_turn_start_time = time.time()

                else:
                    head_status = "Looking Forward"

                    head_turn_start_time = None
                    last_head_warning_time = None

                if head_turn_start_time is not None:
                    head_turn_duration = time.time() - head_turn_start_time
                else:
                    head_turn_duration = 0


                # --------------------------------
                # LEFT EYE
                # --------------------------------

                left_eye_points = []

                for index in LEFT_EYE:

                    landmark = face_landmarks.landmark[index]

                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])

                    left_eye_points.append([x, y])

                    cv2.circle(
                        image,
                        (x, y),
                        3,
                        (0, 0, 255),
                        -1
                    )

                left_eye_points = np.array(left_eye_points)

                left_ear = calculate_ear(left_eye_points)


                # --------------------------------
                # RIGHT EYE
                # --------------------------------

                right_eye_points = []

                for index in RIGHT_EYE:

                    landmark = face_landmarks.landmark[index]

                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])

                    right_eye_points.append([x, y])

                    cv2.circle(
                        image,
                        (x, y),
                        3,
                        (0, 0, 255),
                        -1
                    )

                right_eye_points = np.array(right_eye_points)

                right_ear = calculate_ear(right_eye_points)


                # --------------------------------
                # EYE STATUS
                # --------------------------------

                average_ear = (left_ear + right_ear) / 2

                if average_ear < EAR_THRESHOLD:

                    eye_status = "Closed"

                    if closed_start_time is None:
                        closed_start_time = time.time()

                else:

                    eye_status = "Open"
                    closed_start_time = None
                    last_warning_time = None

                if closed_start_time is not None:
                    closed_duration = time.time() - closed_start_time
                else:
                    closed_duration = 0


                # --------------------------------
                # DROWSINESS WARNING
                # --------------------------------

                if closed_duration >= EYE_CLOSED_TIME:

                    drowsy_warning = True

                    if (
                        last_warning_time is None
                        or time.time() - last_warning_time >= WARNING_INTERVAL
                    ):
                        winsound.Beep(1000, 500)
                        last_warning_time = time.time()

                else:

                    drowsy_warning = False


                # --------------------------------
                # LOOKING-AWAY WARNING
                # --------------------------------

                if head_turn_duration >= HEAD_TURN_TIME:

                    attention_warning = True

                    if (
                        last_head_warning_time is None
                        or time.time() - last_head_warning_time >= WARNING_INTERVAL
                    ):
                        winsound.Beep(1000, 500)
                        last_head_warning_time = time.time()

                else:

                    attention_warning = False


                # --------------------------------
                # MAIN STATUS
                # --------------------------------

                if drowsy_warning:

                    main_status = "DROWSY WARNING"

                elif attention_warning:

                    main_status = "LOOKING AWAY"

                else:

                    main_status = "FOCUSED"


                # --------------------------------
                # UI BACKGROUND
                # --------------------------------

                overlay = image.copy()

                cv2.rectangle(
                    overlay,
                    (15, 15),
                    (image.shape[1] - 15, 370),
                    (30, 30, 30),
                    -1
                )

                image = cv2.addWeighted(
                    overlay,
                    0.75,
                    image,
                    0.25,
                    0
                )


                # --------------------------------
                # TITLE
                # --------------------------------

                cv2.putText(
                    image,
                    "DROWSINESS & FOCUS MONITOR",
                    (40, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    "Real-Time Attention Detection",
                    (40, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1
                )


                # --------------------------------
                # MAIN STATUS
                # --------------------------------

                if main_status == "FOCUSED":

                    status_color = (0, 255, 0)

                else:

                    status_color = (0, 0, 255)

                cv2.putText(
                    image,
                    "STATUS",
                    (40, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 180),
                    1
                )

                cv2.putText(
                    image,
                    main_status,
                    (40, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    status_color,
                    3
                )


                # --------------------------------
                # INFORMATION
                # --------------------------------

                cv2.putText(
                    image,
                    f"Eyes: {eye_status}",
                    (40, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    f"Head: {head_status}",
                    (40, 255),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    f"Eyes Closed: {closed_duration:.1f}s",
                    (40, 295),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 200, 200),
                    1
                )

                cv2.putText(
                    image,
                    f"Looking Away: {head_turn_duration:.1f}s",
                    (40, 325),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 200, 200),
                    1
                )


                # --------------------------------
                # FACE MESH
                # --------------------------------

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=custom_spec,
                    connection_drawing_spec=custom_spec
                )

        else:

            # No face detected
            cv2.putText(
                image,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Drowsiness & Focus Monitor",
            image
        )

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
