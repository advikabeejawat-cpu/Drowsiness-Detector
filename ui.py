import cv2
import textwrap


# ---------------------------------------------------------
# COLORS
# ---------------------------------------------------------

BACKGROUND = (245, 240, 250)
WHITE = (255, 255, 255)

DARK_TEXT = (60, 55, 70)
SOFT_TEXT = (120, 110, 130)

LIGHT_LAVENDER = (235, 225, 250)
LIGHT_PINK = (245, 225, 235)
LIGHT_GREEN = (225, 245, 230)

GREEN = (80, 170, 100)
PINK = (190, 100, 150)
PURPLE = (130, 100, 180)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def draw_text(
    image,
    text,
    position,
    font_scale=0.6,
    color=DARK_TEXT,
    thickness=1
):

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_card(image, x1, y1, x2, y2, color):
    """
    Draw a simple card on the screen.
    """

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )


# ---------------------------------------------------------
# MAIN MONITORING UI
# ---------------------------------------------------------

def draw_ui(
    image,
    main_status,
    calibrated=True,
    prediction_text="",
    prediction_detail="",
    people_statuses=None,
):
    """
    Draw the main Focus Buddy monitoring interface.
    """

    height, width = image.shape[:2]
    people_statuses = people_statuses or []

    # -----------------------------------------------------
    # BACKGROUND
    # -----------------------------------------------------

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, height),
        BACKGROUND,
        -1
    )

    image[:] = cv2.addWeighted(
        overlay,
        0.35,
        image,
        0.65,
        0
    )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    draw_text(
        image,
        "focus buddy",
        (25, 40),
        0.9,
        DARK_TEXT,
        2
    )

    draw_text(
        image,
        "your little attention monitor",
        (27, 65),
        0.45,
        SOFT_TEXT,
        1
    )

    # -----------------------------------------------------
    # CALIBRATION INDICATOR
    # -----------------------------------------------------

    if calibrated:

        draw_text(
            image,
            "PERSONALIZED",
            (width - 125, 40),
            0.4,
            GREEN,
            1
        )

    # -----------------------------------------------------
    # MAIN STATUS CARD
    # -----------------------------------------------------

    draw_card(
        image,
        20,
        85,
        width - 20,
        220,
        WHITE
    )

    draw_text(image, "LIVE MONITORING", (40, 115), 0.45, SOFT_TEXT, 1)

    drowsy_people = [
        person["number"] for person in people_statuses
        if person.get("main_state") == "Drowsy"
    ]
    away_people = [
        person["number"] for person in people_statuses
        if person.get("main_state") == "Looking Away"
    ]
    if drowsy_people:
        status_text = "Eyes closed too long: " + ", ".join(f"Person {n}" for n in drowsy_people)
        status_color = PINK
    elif away_people:
        status_text = "Looking away: " + ", ".join(f"Person {n}" for n in away_people)
        status_color = PURPLE
    elif not people_statuses or main_status == "Tracking Lost":
        status_text = "No faces detected"
        status_color = SOFT_TEXT
    else:
        status_text = f"Monitoring {len(people_statuses)} face" + ("" if len(people_statuses) == 1 else "s")
        status_color = GREEN

    draw_text(
        image,
        status_text,
        (40, 155),
        0.65,
        status_color,
        2
    )

    if prediction_text:
        draw_text(
            image,
            prediction_text,
            (40, 180),
            0.42,
            PURPLE if "Elevated" in prediction_text else SOFT_TEXT,
            1
        )

    if prediction_detail:
        draw_text(
            image,
            prediction_detail,
            (40, 202),
            0.38,
            SOFT_TEXT,
            1
        )

    # -----------------------------------------------------
    # INDIVIDUAL FACE STATUS LIST
    # -----------------------------------------------------

    draw_card(
        image,
        20,
        285,
        width - 20,
        height - 45,
        WHITE
    )

    draw_text(image, f"PEOPLE IN VIEW  ({len(people_statuses)})", (38, 311), 0.46, SOFT_TEXT, 1)
    list_top = 320
    list_bottom = height - 54
    if people_statuses:
        row_height = max(20, (list_bottom - list_top) // len(people_statuses))
        for index, person in enumerate(people_statuses[:5]):
            row_y = list_top + index * row_height
            status = person.get("main_state", "Tracking Lost")
            row_color = LIGHT_PINK if status == "Drowsy" else LIGHT_LAVENDER if status == "Looking Away" else LIGHT_GREEN
            draw_card(image, 30, row_y, width - 30, row_y + row_height - 2, row_color)
            details = [
                f"Person {person['number']}",
                status,
                f"Eyes {person.get('eye_status', 'Unknown')}",
                person.get("head_status", "Unknown"),
            ]
            closed = person.get("closed_duration", 0.0)
            away = person.get("head_turn_duration", 0.0)
            if closed > 0.1:
                details.append(f"Closed {closed:.1f}s")
            if away > 0.1:
                details.append(f"Away {away:.1f}s")
            ear = person.get("average_ear")
            threshold = person.get("eye_threshold")
            if ear is not None and threshold is not None:
                details.append(f"EAR {ear:.2f}/{threshold:.2f}")
            draw_text(image, "  |  ".join(details), (42, row_y + row_height - 6), 0.34, DARK_TEXT, 1)
    else:
        draw_text(image, "No faces detected. Move into the camera view.", (42, 350), 0.48, SOFT_TEXT, 1)

    # -----------------------------------------------------
    # MONITORING INDICATOR
    # -----------------------------------------------------

    cv2.circle(
        image,
        (30, height - 30),
        7,
        GREEN,
        -1
    )

    draw_text(
        image,
        "Monitoring",
        (45, height - 24),
        0.45,
        SOFT_TEXT,
        1
    )

    draw_text(
        image,
        "Press Q to quit",
        (width - 130, height - 24),
        0.4,
        SOFT_TEXT,
        1
    )

    return image


# ---------------------------------------------------------
# CALIBRATION UI
# ---------------------------------------------------------

def draw_calibration_ui(
    image,
    progress,
    remaining_time,
    face_detected
):
    """
    Display the calibration screen.

    progress:
        Calibration progress from 0.0 to 1.0

    remaining_time:
        Seconds remaining in calibration

    face_detected:
        Whether a face is currently visible
    """

    height, width = image.shape[:2]

    # -----------------------------------------------------
    # BACKGROUND
    # -----------------------------------------------------

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, height),
        BACKGROUND,
        -1
    )

    image[:] = cv2.addWeighted(
        overlay,
        0.45,
        image,
        0.55,
        0
    )

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    draw_text(
        image,
        "focus buddy",
        (25, 45),
        0.9,
        DARK_TEXT,
        2
    )

    draw_text(
        image,
        "personal calibration",
        (27, 72),
        0.5,
        SOFT_TEXT,
        1
    )

    # -----------------------------------------------------
    # MAIN CARD
    # -----------------------------------------------------

    draw_card(
        image,
        40,
        120,
        width - 40,
        330,
        WHITE
    )

    draw_text(
        image,
        "Let's learn your normal behavior",
        (70, 170),
        0.65,
        DARK_TEXT,
        2
    )

    draw_text(
        image,
        "Keep your eyes open and look at the camera for 5 seconds",
        (70, 192),
        0.42,
        SOFT_TEXT,
        1
    )

    # -----------------------------------------------------
    # FACE STATUS
    # -----------------------------------------------------

    if face_detected:

        draw_text(
            image,
            "Face detected - stay still",
            (70, 215),
            0.5,
            GREEN,
            1
        )

    else:

        draw_text(
            image,
            "No face detected - look at the camera",
            (70, 215),
            0.5,
            PINK,
            1
        )

    # -----------------------------------------------------
    # PROGRESS BAR
    # -----------------------------------------------------

    bar_x1 = 70
    bar_x2 = width - 70
    bar_y1 = 260
    bar_y2 = 285

    cv2.rectangle(
        image,
        (bar_x1, bar_y1),
        (bar_x2, bar_y2),
        LIGHT_LAVENDER,
        -1
    )

    progress_width = int(
        (bar_x2 - bar_x1) * progress
    )

    if progress_width > 0:

        cv2.rectangle(
            image,
            (bar_x1, bar_y1),
            (bar_x1 + progress_width, bar_y2),
            PURPLE,
            -1
        )

    # -----------------------------------------------------
    # TIME REMAINING
    # -----------------------------------------------------

    draw_text(
        image,
        f"{remaining_time:.1f} seconds remaining",
        (70, 325),
        0.55,
        DARK_TEXT,
        1
    )

    draw_text(
        image,
        "Keep your head centered and eyes open.",
        (70, 365),
        0.4,
        SOFT_TEXT,
        1
    )

    draw_text(
        image,
        "Calibration will automatically finish.",
        (70, 395),
        0.4,
        SOFT_TEXT,
        1
    )

    return image


def draw_session_summary(
    image,
    session_duration,
    logger,
    personal_insight,
    model_summary
):
    """Replace the last camera frame with a readable end-of-session report."""
    height, width = image.shape[:2]
    image[:] = BACKGROUND

    draw_text(image, "focus buddy", (28, 42), 0.9, DARK_TEXT, 2)
    draw_text(image, "SESSION SUMMARY", (30, 78), 0.55, PURPLE, 2)

    minutes = int(session_duration // 60)
    seconds = int(session_duration % 60)
    summary_rows = [
        ("Session", f"{minutes:02d}:{seconds:02d}  |  {len(logger.events)} recorded events"),
        ("Blinks", str(logger.count_events("Blink"))),
        (
            "Prolonged eye closures",
            f"{logger.count_events('Prolonged Eye Closure')}  |  "
            f"{logger.total_duration('Prolonged Eye Closure'):.1f}s total"
        ),
        (
            "Looking away",
            f"{logger.count_events('Looking Away')}  |  "
            f"{logger.total_duration('Looking Away'):.1f}s total"
        ),
        (
            "Tracking lost",
            f"{logger.count_events('Tracking Lost')}  |  "
            f"{logger.total_duration('Tracking Lost'):.1f}s total"
        ),
    ]

    y = 112
    for label, value in summary_rows:
        draw_card(image, 25, y - 22, width - 25, y + 12, WHITE)
        draw_text(image, label, (40, y), 0.43, SOFT_TEXT, 1)
        draw_text(image, value, (width // 2, y), 0.43, DARK_TEXT, 1)
        y += 42

    section_y = y + 4
    draw_text(image, "PERSONAL LEARNING", (30, section_y), 0.5, PURPLE, 2)
    for index, line in enumerate(textwrap.wrap(model_summary, width=68)[:2]):
        draw_text(image, line, (32, section_y + 27 + index * 22), 0.4, DARK_TEXT, 1)

    insight_y = section_y + 78
    draw_text(image, "YOUR FOCUS INSIGHT", (30, insight_y), 0.5, PURPLE, 2)
    for index, line in enumerate(textwrap.wrap(personal_insight, width=68)[:2]):
        draw_text(image, line, (32, insight_y + 27 + index * 22), 0.4, DARK_TEXT, 1)

    draw_text(image, "Press any key to close", (30, height - 20), 0.4, SOFT_TEXT, 1)
    return image
