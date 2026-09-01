import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# face coordinates for left eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# face coordinates for right eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20
EYE_CLOSED_TIME = 2.0

closed_start_time = None


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

                # Get the points for the left eye
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

                # Calculate the left eye EAR
                left_ear = calculate_ear(left_eye_points)

                # Get the points for the right eye
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

                # Calculate the right eye EAR
                right_ear = calculate_ear(right_eye_points)


                # Average the two eyes
                average_ear = (left_ear + right_ear) / 2


                # Decide whether the eyes are open or closed
                if average_ear < EAR_THRESHOLD:

                    eye_status = "Closed"

                    if closed_start_time is None:
                        closed_start_time = time.time()

                else:

                    eye_status = "Open"
                    closed_start_time = None


                if closed_start_time is not None:
                    closed_duration = time.time() - closed_start_time
                else:
                    closed_duration = 0

                if closed_duration >= EYE_CLOSED_TIME:
                    status = "Drowsy Warning"
                else:
                    status = "Focused"


                cv2.putText(
                    image,
                    f"EAR: {average_ear:.2f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"Eyes: {eye_status}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                status_color = (
                    (0, 0, 255)
                    if status == "Drowsy Warning"
                    else (0, 255, 0)
                )

                cv2.putText(
                    image,
                    f"Status: {status}",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    status_color,
                    2
                )

                cv2.putText(
                    image,
                    f"Closed: {closed_duration:.1f}s",
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=custom_spec,
                    connection_drawing_spec=custom_spec
                )

        cv2.imshow(
            "MediaPipe Face Mesh",
            image
        )

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()