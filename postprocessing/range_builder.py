from copy import deepcopy


class RangeBuilder:
    """
    Converts frame-level detections into
    continuous time ranges.

    Input:

    {
        "objects": {
            "product.jpg": [
                {
                    "time": 1,
                    "bbox": [...]
                },
                {
                    "time": 2,
                    "bbox": [...]
                }
            ]
        }
    }

    Output:

    {
        "objects": {
            "product.jpg": [
                {
                    "start": 1,
                    "end": 3,
                    "bbox": [...]
                }
            ]
        }
    }
    """

    @staticmethod
    def _build_track_ranges(track):

        track = sorted(
            track,
            key=lambda x: x["time"]
        )

        if not track:
            return []

        ranges = []

        start_time = int(track[0]["time"])
        prev_time = int(track[0]["time"])

        current_bbox = deepcopy(
            track[0]["bbox"]
        )
        # Optional: the description of the detection that starts the range.
        current_description = track[0].get("description")

        for item in track[1:]:

            curr_time = int(item["time"])
            curr_bbox = item["bbox"]

            should_split = (
                (curr_time - prev_time > 1)
                or
                (curr_bbox != current_bbox)
            )

            if should_split:

                ranges.append(
                    RangeBuilder._range(
                        start_time, prev_time + 1, current_bbox, current_description
                    )
                )

                start_time = curr_time
                current_bbox = deepcopy(curr_bbox)
                current_description = item.get("description")

            prev_time = curr_time

        ranges.append(
            RangeBuilder._range(
                start_time, prev_time + 1, current_bbox, current_description
            )
        )

        return ranges

    @staticmethod
    def _range(start, end, bbox, description=None):

        entry = {
            "start": start,
            "end": end,
            "bbox": bbox
        }

        if description is not None:
            entry["description"] = deepcopy(description)

        return entry

    def build_episode(
        self,
        episode_output: dict
    ):

        result = {
            "objects": {},
            "clothing": {}
        }

        for category in [
            "objects",
            "clothing"
        ]:

            detections = episode_output.get(
                category,
                {}
            )

            for label, track in detections.items():

                result[category][label] = (
                    self._build_track_ranges(track)
                )

        return result