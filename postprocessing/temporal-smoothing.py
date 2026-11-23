import json
import os

# ✅ Input file (your path)
INPUT_FILE = r"outputs\video\video_timestamps.json"

# ✅ Output file (same folder)
OUTPUT_FILE = os.path.join(
    os.path.dirname(INPUT_FILE),
    "output_ranges.json"
)


# ---------------------------------------
# Convert track → time ranges
# ---------------------------------------
def process_track_ranges(track):
    """
    Converts per-second detections into time ranges
    Example:
    1 → A, 2 → A, 3 → B
    =>
    [1-3 → A], [3-4 → B]
    """

    # Sort by time
    track = sorted(track, key=lambda x: x["time"])
    result = []

    if not track:
        return result

    # Initialize
    start_time = int(track[0]["time"])
    prev_time = start_time
    current_bbox = track[0]["bbox"]

    for i in range(1, len(track)):
        curr = track[i]
        curr_time = int(curr["time"])
        curr_bbox = curr["bbox"]

        # 🚨 Condition to break segment
        if (curr_time - prev_time > 1) or (curr_bbox != current_bbox):
            result.append({
                "start": start_time,
                "end": prev_time + 1,
                "bbox": current_bbox
            })

            # Start new segment
            start_time = curr_time
            current_bbox = curr_bbox

        prev_time = curr_time

    # अंतिम segment
    result.append({
        "start": start_time,
        "end": prev_time + 1,
        "bbox": current_bbox
    })

    return result


# ---------------------------------------
# Process full JSON
# ---------------------------------------
def process_data(data):
    output = {}

    for category in data:
        output[category] = {}

        for label, track in data[category].items():
            output[category][label] = process_track_ranges(track)

    return output


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    print("📥 Loading input JSON...")
    
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    print("⚙️ Processing temporal ranges...")

    output = process_data(data)

    print("💾 Saving output...")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=4)

    print(f"✅ Done! Output saved at:\n{OUTPUT_FILE}")