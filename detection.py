# ==========================================
# DETECTION
# ==========================================
#
# This file handles the computer-vision
# measurements extracted from MediaPipe.
#
# It handles:
#
#   - Eye Aspect Ratio (EAR)
#   - Eye status
#   - Head position
#   - Head-position smoothing
#
# It does NOT decide how long an event lasts.
# That is handled by state_machine.py.
# ==========================================

import numpy as np

from settings import (
    EAR_THRESHOLD,
    HEAD_TURN_THRESHOLD1,
    HEAD_TURN_THRESHOLD2,
    SMOOTHING_FRAMES
)


# ==========================================
# FACIAL LANDMARKS
# ==========================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

NOSE = 1

LEFT_FACE = 234
RIGHT_FACE = 454


# ==========================================
# CALCULATE EAR
# ==========================================

def calculate_ear(eye_points):
    """
    Calculates Eye Aspect Ratio.

    A larger EAR means the eye is more open.
    A smaller EAR means the eye is more closed.
    """

    vertical_1 = np.linalg.norm(
        eye_points[1] - eye_points[5]
    )

    vertical_2 = np.linalg.norm(
        eye_points[2] - eye_points[4]
    )

    horizontal = np.linalg.norm(
        eye_points[0] - eye_points[3]
    )

    # Prevent division by zero.
    if horizontal == 0:
        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )

    return ear


# ==========================================
# GET EYE POINTS
# ==========================================

def get_eye_points(
    face_landmarks,
    eye_landmarks,
    image_width,
    image_height
):
    """
    Converts MediaPipe's normalized landmark
    coordinates into pixel coordinates.
    """

    # Keep subpixel landmark precision. Rounding coordinates to whole pixels
    # can distort the small vertical distances used by EAR, especially on a
    # lower-resolution webcam feed.
    points = np.array([
        [
            face_landmarks.landmark[index].x * image_width,
            face_landmarks.landmark[index].y * image_height,
        ]

        for index in eye_landmarks
    ], dtype=np.float32)

    return points


# ==========================================
# GET EYE STATUS
# ==========================================

def get_eye_status(
    face_landmarks,
    image_width,
    image_height,
    eye_threshold=None
):
    """
    Determines whether the eyes are open
    or closed.

    Returns:
        eye_status
        average_ear
        left_eye_points
        right_eye_points
    """

    # --------------------------------------
    # Get left-eye landmarks
    # --------------------------------------

    left_eye_points = get_eye_points(
        face_landmarks,
        LEFT_EYE,
        image_width,
        image_height
    )


    # --------------------------------------
    # Get right-eye landmarks
    # --------------------------------------

    right_eye_points = get_eye_points(
        face_landmarks,
        RIGHT_EYE,
        image_width,
        image_height
    )


    # --------------------------------------
    # Calculate EAR for each eye
    # --------------------------------------

    left_ear = calculate_ear(
        left_eye_points
    )

    right_ear = calculate_ear(
        right_eye_points
    )


    # Average the two eyes.
    average_ear = (
        left_ear + right_ear
    ) / 2.0


    # --------------------------------------
    # Choose threshold
    # --------------------------------------
    #
    # If calibration has produced a
    # personalized threshold, face.py
    # passes it here.
    #
    # Otherwise, use the default setting.
    # --------------------------------------

    if eye_threshold is None:

        eye_threshold = EAR_THRESHOLD


    # --------------------------------------
    # Determine eye state
    # --------------------------------------

    if average_ear < eye_threshold:

        eye_status = "Closed"

    else:

        eye_status = "Open"


    return (
        eye_status,
        average_ear,
        left_eye_points,
        right_eye_points
    )


# ==========================================
# GET HEAD POSITION
# ==========================================

def get_head_position(
    face_landmarks,
    head_positions,
    head_thresholds=None
):
    """
    Determines whether the user is looking
    forward or away.

    The most recent measurements are averaged
    to reduce sudden frame-to-frame changes.
    """

    # --------------------------------------
    # Get important landmarks
    # --------------------------------------

    nose = face_landmarks.landmark[NOSE]

    left_face = (
        face_landmarks.landmark[LEFT_FACE]
    )

    right_face = (
        face_landmarks.landmark[RIGHT_FACE]
    )


    # --------------------------------------
    # Calculate face width
    # --------------------------------------

    face_width = (
        right_face.x - left_face.x
    )


    if face_width == 0:

        return (
            "Unknown",
            0.5,
            head_positions
        )


    # --------------------------------------
    # Calculate normalized nose position
    # --------------------------------------

    nose_position = (
        nose.x - left_face.x
    ) / face_width


    # --------------------------------------
    # Store recent measurement
    # --------------------------------------

    head_positions.append(
        nose_position
    )


    # Keep only the most recent frames.
    if len(head_positions) > SMOOTHING_FRAMES:

        head_positions.pop(0)


    # --------------------------------------
    # Smooth measurement
    # --------------------------------------

    smoothed_position = (
        sum(head_positions)
        / len(head_positions)
    )


    # --------------------------------------
    # Choose thresholds
    # --------------------------------------

    if head_thresholds is None:

        low_threshold = HEAD_TURN_THRESHOLD1
        high_threshold = HEAD_TURN_THRESHOLD2

    else:

        low_threshold, high_threshold = (
            head_thresholds
        )


    # --------------------------------------
    # Determine head state
    # --------------------------------------

    if (
        smoothed_position < low_threshold
        or
        smoothed_position > high_threshold
    ):

        head_status = "Looking Away"

    else:

        head_status = "Looking Forward"


    return (
        head_status,
        smoothed_position,
        head_positions
    )
