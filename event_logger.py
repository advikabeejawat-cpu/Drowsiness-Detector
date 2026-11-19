# ==========================================
# EVENT LOGGER
# ==========================================
# This file records important events that
# happen during a monitoring session.
#
# Examples:
#   - Blink
#   - Prolonged eye closure
#   - Looking away
#   - Camera tracking interruption
#
# Later, this data can be used to create
# session summaries, graphs, and comparisons.
# ==========================================

import csv
from datetime import datetime


class EventLogger:

    def __init__(self):
        # List containing every event during
        # the current session.
        self.events = []

        # Time when the session started.
        self.session_start = datetime.now()


    # ======================================
    # RECORD AN EVENT
    # ======================================

    def log_event(self, event_type, start_time, end_time):
        """
        Records one completed event.

        event_type:
            What happened.

        start_time:
            Seconds since the session started.

        end_time:
            Seconds since the session started.
        """

        duration = end_time - start_time

        event = {
            "type": event_type,
            "start": start_time,
            "end": end_time,
            "duration": duration
        }

        self.events.append(event)


    # ======================================
    # COUNT EVENTS
    # ======================================

    def count_events(self, event_type):
        """
        Returns how many times a specific
        event occurred.
        """

        count = 0

        for event in self.events:

            if event["type"] == event_type:
                count += 1

        return count


    # ======================================
    # TOTAL DURATION
    # ======================================

    def total_duration(self, event_type):
        """
        Returns the total amount of time spent
        in a specific event.
        """

        total = 0

        for event in self.events:

            if event["type"] == event_type:
                total += event["duration"]

        return total


    # ======================================
    # LONGEST EVENT
    # ======================================

    def longest_event(self, event_type):
        """
        Returns the longest occurrence of a
        specific event.
        """

        durations = []

        for event in self.events:

            if event["type"] == event_type:
                durations.append(event["duration"])

        if len(durations) == 0:
            return 0

        return max(durations)


    # ======================================
    # AVERAGE EVENT
    # ======================================

    def average_event(self, event_type):
        """
        Returns the average duration of an event.
        """

        durations = []

        for event in self.events:

            if event["type"] == event_type:
                durations.append(event["duration"])

        if len(durations) == 0:
            return 0

        return sum(durations) / len(durations)


    # ======================================
    # SAVE DATA TO CSV
    # ======================================

    def save_csv(self):
        """
        Saves all session events to a CSV file.

        Example:
        session_2026-09-30_143500.csv
        """

        timestamp = self.session_start.strftime("%Y-%m-%d_%H%M%S")

        filename = f"session_{timestamp}.csv"

        with open(
            filename,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            # Column names
            writer.writerow([
                "Event Type",
                "Start Time",
                "End Time",
                "Duration"
            ])

            # Write every event
            for event in self.events:

                writer.writerow([
                    event["type"],
                    round(event["start"], 2),
                    round(event["end"], 2),
                    round(event["duration"], 2)
                ])

        return filename


    # ======================================
    # PRINT SESSION SUMMARY
    # ======================================

    def print_summary(self, session_duration):
        """
        Prints a detailed summary when the
        monitoring session ends.
        """

        print("\n")
        print("=" * 50)
        print("             SESSION SUMMARY")
        print("=" * 50)

        # ----------------------------------
        # Session information
        # ----------------------------------

        minutes = int(session_duration // 60)

        seconds = int(session_duration % 60)

        print(
            f"\nSession duration: {minutes:02d}:{seconds:02d}"
        )


        # ----------------------------------
        # Eye information
        # ----------------------------------

        blink_count = self.count_events("Blink")

        prolonged_count = self.count_events(
            "Prolonged Eye Closure"
        )

        total_closed = self.total_duration(
            "Prolonged Eye Closure"
        )

        longest_closed = self.longest_event(
            "Prolonged Eye Closure"
        )


        print("\nEYE ACTIVITY")
        print("-" * 50)

        print(
            f"Blink events:             {blink_count}"
        )

        print(
            f"Prolonged closures:       {prolonged_count}"
        )

        print(
            f"Total prolonged closure:  {total_closed:.1f}s"
        )

        print(
            f"Longest eye closure:      {longest_closed:.1f}s"
        )


        # ----------------------------------
        # Attention information
        # ----------------------------------

        distraction_count = self.count_events(
            "Looking Away"
        )

        total_away = self.total_duration(
            "Looking Away"
        )

        longest_away = self.longest_event(
            "Looking Away"
        )

        average_away = self.average_event(
            "Looking Away"
        )


        print("\nATTENTION")
        print("-" * 50)

        print(
            f"Looking-away events:      {distraction_count}"
        )

        print(
            f"Total looking-away time:  {total_away:.1f}s"
        )

        print(
            f"Longest distraction:      {longest_away:.1f}s"
        )

        print(
            f"Average distraction:      {average_away:.1f}s"
        )


        # ----------------------------------
        # Tracking information
        # ----------------------------------

        tracking_count = self.count_events(
            "Tracking Lost"
        )

        tracking_time = self.total_duration(
            "Tracking Lost"
        )


        print("\nCAMERA TRACKING")
        print("-" * 50)

        print(
            f"Tracking interruptions:   {tracking_count}"
        )

        print(
            f"Time without tracking:     {tracking_time:.1f}s"
        )


        # ----------------------------------
        # Overall event count
        # ----------------------------------

        print("\nSESSION EVENTS")
        print("-" * 50)

        print(
            f"Total recorded events:    {len(self.events)}"
        )

        print("\n" + "=" * 50)