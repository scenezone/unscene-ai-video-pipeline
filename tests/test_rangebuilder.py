from postprocessing.range_builder import RangeBuilder

sample = {
    "objects": {
        "watch.jpg": [
            {
                "time": 1,
                "bbox": [10, 10, 20, 20]
            },
            {
                "time": 2,
                "bbox": [10, 10, 20, 20]
            },
            {
                "time": 5,
                "bbox": [10, 10, 20, 20]
            }
        ]
    },
    "clothing": {}
}

builder = RangeBuilder()

result = builder.build(sample)

print(result)