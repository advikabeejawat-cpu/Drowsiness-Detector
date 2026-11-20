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

        # Keep every valid measurement. The baseline calculation below uses
        # the upper portion of the samples so brief blinks/closures do not
        # dominate, without rejecting naturally low EAR values at a fixed
        # default cutoff.
        if ear is not None and ear > 0:

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

        self.baseline_ear = self.get_eye_baseline()
        if self.baseline_ear is not None:
            self.eye_threshold = self._threshold_from_baseline(self.baseline_ear)


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

    def get_eye_baseline(self):
        """Estimate open-eye EAR from the higher, likely-open samples."""
        if len(self.ear_values) < 10:
            return None
        ordered = sorted(self.ear_values)
        open_samples = ordered[int(len(ordered) * 0.40):]
        return statistics.median(open_samples)

    @staticmethod
    def _threshold_from_baseline(baseline):
        return min(0.30, max(0.12, baseline * 0.70))

    def get_eye_threshold_preview(self):
        baseline = self.get_eye_baseline()
        if baseline is None:
            return None
        return self._threshold_from_baseline(baseline)


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
            "eye_threshold_preview": self.get_eye_threshold_preview(),
            "baseline_head": self.baseline_head,
            "head_low": self.head_low,
            "head_high": self.head_high,
            "samples": len(self.ear_values)
        }
