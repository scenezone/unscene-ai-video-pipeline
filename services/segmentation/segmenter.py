import numpy as np


class SAMSegmenter:

    def __init__(self, predictor):
        self.predictor = predictor

    def segment(self, frame, bbox):
        segmented, _ = self.segment_with_area_ratio(frame, bbox)
        return segmented

    def segment_with_area_ratio(self, frame, bbox):

        x1, y1, x2, y2 = bbox

        input_box = np.array([x1, y1, x2, y2])

        masks, _, _ = self.predictor.predict(
            box=input_box,
            multimask_output=False
        )

        mask = masks[0]

        segmented = frame.copy()
        segmented[~mask] = 0

        mask_crop = mask[y1:y2, x1:x2]
        area_ratio = float(np.count_nonzero(mask_crop) / mask_crop.size) if mask_crop.size else 0.0
        return segmented[y1:y2, x1:x2], area_ratio