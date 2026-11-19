"""Learn a personal short-term distraction predictor from saved session CSVs.

The training target is whether a look-away or prolonged eye-closure event
starts in the next 10 seconds. Recent look-away, prolonged-closure, and
tracking-loss events provide context; normal blinks are excluded.
"""

import csv
import glob
import math
import os


BUCKET_SECONDS = 5
HISTORY_SECONDS = 15
PREDICTION_SECONDS = 10
DISTRACTION_TYPES = {"Looking Away", "Prolonged Eye Closure"}
FEATURE_TYPES = {
    "Looking Away": "away",
    "Prolonged Eye Closure": "eye_closed",
    "Tracking Lost": "tracking",
}
FEATURES = ("away", "eye_closed", "tracking")
MIN_TRAINING_WINDOWS = 12
MIN_POSITIVE_WINDOWS = 3
MIN_NEGATIVE_WINDOWS = 3


class BehaviorPredictor:
    def __init__(self, directory=None):
        self.directory = directory or os.path.dirname(os.path.abspath(__file__))
        self.model = None
        self.sample_count = 0
        self.positive_count = 0
        self.retrain()

    @staticmethod
    def _read_session(path):
        events = []
        session_duration = None
        try:
            with open(path, newline="", encoding="utf-8-sig") as source:
                for row in csv.DictReader(source):
                    kind = (row.get("Event Type") or "").strip()
                    if kind == "Session Duration":
                        try:
                            session_duration = max(0.0, float(row.get("Duration", "")))
                        except (TypeError, ValueError):
                            pass
                        continue
                    try:
                        start = float(row.get("Start Time", ""))
                        end = float(row.get("End Time", ""))
                        duration = float(row.get("Duration", ""))
                    except (TypeError, ValueError):
                        continue
                    if kind in (*FEATURE_TYPES, "Blink") and start >= 0 and end >= start:
                        events.append((kind, start, end, max(0.0, duration)))
        except (OSError, csv.Error):
            return [], None
        return events, session_duration

    @classmethod
    def _read_events(cls, path):
        """Compatibility helper for code that needs only event rows."""
        events, _session_duration = cls._read_session(path)
        return events

    @staticmethod
    def _features(events, at_time):
        start = at_time - HISTORY_SECONDS
        counts = {feature: 0.0 for feature in FEATURES}
        for event in events:
            if isinstance(event, dict):
                kind = event.get("type")
                event_start = event.get("start")
                event_end = event.get("end")
            else:
                kind, event_start, event_end = event[:3]
            try:
                event_start = float(event_start)
                event_end = float(event_end)
            except (TypeError, ValueError):
                continue
            if event_start <= at_time and event_end >= start:
                feature_name = FEATURE_TYPES.get(kind)
                if feature_name is not None:
                    counts[feature_name] += 1.0
        return counts

    @classmethod
    def _examples(cls, events, session_duration=None):
        if not events and session_duration is None:
            return []
        # New logs include the session boundary, allowing quiet/no-event
        # intervals to count as negatives. Legacy files have no boundary, so
        # conservatively stop before their last observed event.
        observed_until = (
            session_duration
            if session_duration is not None
            else max(event[2] for event in events)
        )
        last = observed_until - PREDICTION_SECONDS
        samples = []
        first_index = int(HISTORY_SECONDS // BUCKET_SECONDS)
        for index in range(first_index, int(last // BUCKET_SECONDS) + 1):
            at_time = index * BUCKET_SECONDS
            features = cls._features(events, at_time)
            future_distraction = any(
                kind in DISTRACTION_TYPES and at_time <= start < at_time + PREDICTION_SECONDS
                for kind, start, _end, _duration in events
            )
            samples.append((features, int(future_distraction)))
        return samples

    def retrain(self, additional_events=None):
        examples = []
        paths = glob.glob(os.path.join(self.directory, "**", "session_*.csv"), recursive=True)
        paths = [
            path for path in paths
            if not any(part.lower() in {".git", ".venv", "venv", "__pycache__"}
                       for part in os.path.normpath(path).split(os.sep))
        ]
        for path in paths:
            events, session_duration = self._read_session(path)
            examples.extend(self._examples(events, session_duration))
        if additional_events:
            examples.extend(self._examples(additional_events))
        self.sample_count = len(examples)
        self.positive_count = sum(label for _features, label in examples)
        negative_count = self.sample_count - self.positive_count
        if (
            self.sample_count < MIN_TRAINING_WINDOWS
            or self.positive_count < MIN_POSITIVE_WINDOWS
            or negative_count < MIN_NEGATIVE_WINDOWS
        ):
            self.model = None
            return

        negatives = negative_count
        # Gaussian Naive Bayes with variance smoothing for sparse count features.
        model = {}
        for label, count in ((0, negatives), (1, self.positive_count)):
            class_rows = [features for features, target in examples if target == label]
            stats = {}
            for feature in FEATURES:
                values = [row[feature] for row in class_rows]
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                stats[feature] = (mean, max(variance, 0.25))
            model[label] = (count / self.sample_count, stats)
        self.model = model

    def predict(self, elapsed, events):
        if self.model is None:
            return {"ready": False, "probability": None, "samples": self.sample_count}
        features = self._features(events, elapsed)
        log_scores = {}
        for label, (prior, stats) in self.model.items():
            score = math.log(max(prior, 1e-9))
            for feature, value in features.items():
                mean, variance = stats[feature]
                score -= 0.5 * (math.log(2 * math.pi * variance) + ((value - mean) ** 2 / variance))
            log_scores[label] = score
        offset = max(log_scores.values())
        p0 = math.exp(log_scores[0] - offset)
        p1 = math.exp(log_scores[1] - offset)
        probability = p1 / (p0 + p1)
        driver = max(
            FEATURES,
            key=lambda feature: (
                self.model[1][1][feature][0]
                - self.model[0][1][feature][0]
            ) * features[feature]
        )
        driver_label = {
            "away": "recent looking-away events",
            "eye_closed": "recent prolonged eye-closure events",
            "tracking": "recent tracking interruptions",
        }[driver]
        return {
            "ready": True,
            "probability": probability,
            "samples": self.sample_count,
            "driver": driver_label if features[driver] else "no recent warning events",
        }

    def session_insight(self, current_events):
        """Compare this session's away episodes with the user's history."""
        historical_durations = []
        paths = glob.glob(os.path.join(self.directory, "**", "session_*.csv"), recursive=True)
        for path in paths:
            historical_durations.extend(
                duration for kind, _start, _end, duration in self._read_events(path)
                if kind == "Looking Away" and duration > 0
            )

        current_durations = []
        for event in current_events:
            if isinstance(event, dict) and event.get("type") == "Looking Away":
                try:
                    duration = float(event.get("duration", 0))
                except (TypeError, ValueError):
                    continue
                if duration > 0:
                    current_durations.append(duration)

        if not historical_durations:
            return "Not enough past look-away events yet to build your personal baseline."
        if not current_durations:
            typical = sum(historical_durations) / len(historical_durations)
            return (
                f"Your past look-away episodes average {typical:.1f}s; "
                "no completed look-away episodes this session."
            )

        typical = sum(historical_durations) / len(historical_durations)
        current = sum(current_durations) / len(current_durations)
        delta = current - typical
        if abs(delta) < 0.2:
            comparison = "about the same as your usual pattern"
        elif delta < 0:
            comparison = f"{abs(delta):.1f}s shorter than your usual pattern"
        else:
            comparison = f"{delta:.1f}s longer than your usual pattern"
        return (
            f"Your look-away episodes averaged {current:.1f}s this session, "
            f"{comparison} ({typical:.1f}s across past sessions)."
        )

