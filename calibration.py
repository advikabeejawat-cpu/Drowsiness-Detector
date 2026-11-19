# ==========================================
# PERSONAL CALIBRATION
# ==========================================
#
# Learns the user's normal:
#
#   - Eye Aspect Ratio
#   - Head position
#
# The detector then uses these measurements
# to create personalized thresholds.
# ==========================================

import time
import statistics

from settings import (
    CALIBRATION_TIME,
    EAR_THRESHOLD,
    HEAD_TURN_THRESHOLD1,
    HEAD_TURN_THRESHOLD2
)


class Calibration:

    def __init__(self):

        self.active = False
        self.complete = False

        self.start_time = None

        # Raw measurements collected during
        # calibration.
        self.ear_values = []
        self.head_values = []

        # Personalized eye settings.
        self.baseline_ear = None
        self.eye_threshold = EAR_THRESHOLD

        # Personalized head settings.
        self.baseline_head = None
        self.head_low = HEAD_TURN_THRESHOLD1
        self.head_high = HEAD_TURN_THRESHOLD2


    # ======================================
    # START
    # ======================================

    def start(self):

        self.active = True
        self.complete = False

        self.start_time = time.time()

        self.ear_values.clear()
        self.head_values.clear()


    # ======================================
    # UPDATE
    # ======================================

    def update(self, ear, head_position):

        if not self.active:
            return

        # Calibration should represent the user's open-eye baseline. A low EAR
        # sample is likely a blink or a closed-eye frame and would otherwise
        # pull the personalized threshold down enough to miss later closures.
        if ear is not None and ear >= EAR_THRESHOLD:

            self.ear_values.append(ear)

        if head_position is not None:

            self.head_values.append(
                head_position
            )

        elapsed = (
            time.time() - self.start_time
        )

        if elapsed >= CALIBRATION_TIME:

            self.finish()


    # ======================================
    # FINISH
    # ======================================

    def finish(self):

        self.active = False
        self.complete = True


        # ----------------------------------
        # EYE CALIBRATION
        # ----------------------------------

        if self.ear_values:

            self.baseline_ear = statistics.median(
                self.ear_values
            )

            self.eye_threshold = min(
                0.30,
                max(0.18, self.baseline_ear * 0.70),
            )


        # ----------------------------------
        # HEAD CALIBRATION
        # ----------------------------------

        if self.head_values:

            self.baseline_head = statistics.median(
                self.head_values
            )

            tolerance = 0.18

            self.head_low = max(
                0.15,
                self.baseline_head - tolerance
            )

            self.head_high = min(
                0.85,
                self.baseline_head + tolerance
            )


    # ======================================
    # PROGRESS
    # ======================================

    def get_progress(self):

        if not self.active:

            if self.complete:
                return 1.0

            return 0.0

        elapsed = (
            time.time() - self.start_time
        )

        return min(
            elapsed / CALIBRATION_TIME,
            1.0
        )


    # ======================================
    # REMAINING TIME
    # ======================================

    def get_remaining_time(self):

        if not self.active:
            return 0.0

        elapsed = (
            time.time() - self.start_time
        )

        return max(
            CALIBRATION_TIME - elapsed,
            0.0
        )


    # ======================================
    # EYE THRESHOLD
    # ======================================

    def get_eye_threshold(self):

        return self.eye_threshold


    # ======================================
    # HEAD THRESHOLDS
    # ======================================

    def get_head_thresholds(self):

        return (
            self.head_low,
            self.head_high
        )


    # ======================================
    # SUMMARY
    # ======================================

    def get_summary(self):

        return {
            "baseline_ear": self.baseline_ear,
            "eye_threshold": self.eye_threshold,
            "baseline_head": self.baseline_head,
            "head_low": self.head_low,
            "head_high": self.head_high,
            "samples": len(self.ear_values)
        }
