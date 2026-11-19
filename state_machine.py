import time

from settings import EYE_CLOSED_TIME, HEAD_TURN_TIME


class DetectionState:
    """
    Controls the temporary states of the detector.

    Instead of reacting to a single frame, this class watches
    what happens over time.

    Example:
        Eyes closed for 0.3 seconds -> probably a blink
        Eyes closed for 2+ seconds -> prolonged eye closure

        Looking away for 0.5 seconds -> probably normal movement
        Looking away for 2+ seconds -> distraction warning
    """

    def __init__(self):
        # Current states
        self.eye_state = "Open"
        self.head_state = "Centered"
        self.main_state = "Focused"

        # When an event started
        self.eye_closed_start = None
        self.head_away_start = None

        # Whether a warning is currently active
        self.drowsiness_active = False
        self.distraction_active = False

    def update(self, eye_status, head_status):
        """
        Update the state machine using the newest detection results.

        Returns a dictionary containing everything the UI and
        event logger need.
        """

        current_time = time.time()

        events = []

        # ---------------------------------------------------------
        # EYE STATE
        # ---------------------------------------------------------

        if eye_status == "Closed":

            # Start the timer the first frame the eyes close
            if self.eye_closed_start is None:
                self.eye_closed_start = current_time

            closed_duration = current_time - self.eye_closed_start

            # Eyes have been closed long enough to be concerning
            if closed_duration >= EYE_CLOSED_TIME:
                self.drowsiness_active = True

        else:

            # Eyes just opened
            if self.eye_closed_start is not None:

                closed_duration = current_time - self.eye_closed_start

                # Short closure = blink
                if closed_duration < EYE_CLOSED_TIME:
                    events.append({
                        "type": "Blink",
                        "start": self.eye_closed_start,
                        "end": current_time,
                        "duration": closed_duration
                    })

                # Long closure = prolonged eye closure
                else:
                    events.append({
                        "type": "Prolonged Eye Closure",
                        "start": self.eye_closed_start,
                        "end": current_time,
                        "duration": closed_duration
                    })

                # Reset eye timer
                self.eye_closed_start = None

            # Eyes are open again, so drowsiness warning ends
            self.drowsiness_active = False

        # ---------------------------------------------------------
        # HEAD / ATTENTION STATE
        # ---------------------------------------------------------

        if head_status == "Looking Away":

            # Start the timer the first frame the person looks away
            if self.head_away_start is None:
                self.head_away_start = current_time

            head_away_duration = current_time - self.head_away_start

            # Looking away long enough to be concerning
            if head_away_duration >= HEAD_TURN_TIME:
                self.distraction_active = True

        else:

            # Person has returned to the centered position
            if self.head_away_start is not None:

                head_away_duration = current_time - self.head_away_start

                # Only record it as an event if it lasted long enough
                if head_away_duration >= HEAD_TURN_TIME:
                    events.append({
                        "type": "Looking Away",
                        "start": self.head_away_start,
                        "end": current_time,
                        "duration": head_away_duration
                    })

                # Reset head timer
                self.head_away_start = None

            # Person is no longer looking away
            self.distraction_active = False

        # ---------------------------------------------------------
        # DETERMINE MAIN STATE
        # ---------------------------------------------------------

        if self.drowsiness_active:
            self.main_state = "Drowsy"

        elif self.distraction_active:
            self.main_state = "Looking Away"

        elif eye_status == "No Face" or head_status == "No Face":
            self.main_state = "Tracking Lost"

        else:
            self.main_state = "Focused"

        # ---------------------------------------------------------
        # GET CURRENT DURATIONS
        # ---------------------------------------------------------

        if self.eye_closed_start is not None:
            closed_duration = current_time - self.eye_closed_start
        else:
            closed_duration = 0.0

        if self.head_away_start is not None:
            head_away_duration = current_time - self.head_away_start
        else:
            head_away_duration = 0.0

        # ---------------------------------------------------------
        # RETURN EVERYTHING
        # ---------------------------------------------------------

        return {
            "main_state": self.main_state,

            "eye_status": eye_status,
            "head_status": head_status,

            "closed_duration": closed_duration,
            "head_turn_duration": head_away_duration,

            "drowsy_warning": self.drowsiness_active,
            "attention_warning": self.distraction_active,

            "events": events
        }

    def get_closed_duration(self):
        """Return how long the eyes have currently been closed."""

        if self.eye_closed_start is None:
            return 0.0

        return time.time() - self.eye_closed_start

    def get_head_turn_duration(self):
        """Return how long the person has currently been looking away."""

        if self.head_away_start is None:
            return 0.0

        return time.time() - self.head_away_start

    def reset(self):
        """Reset the state machine."""

        self.eye_state = "Open"
        self.head_state = "Centered"
        self.main_state = "Focused"

        self.eye_closed_start = None
        self.head_away_start = None

        self.drowsiness_active = False
        self.distraction_active = False