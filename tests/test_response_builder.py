import json
from pathlib import Path
import sys

# Project root
sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)

from postprocessing.response_builder import ResponseBuilder


# --------------------------------------------------
# MOCK JOB SERVICE OUTPUT
# --------------------------------------------------

job_result = {
    "jobId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "seriesId": "69fe36c51f3977b48b2e7782",

    "results": [
        {
            "episodeId": "episode_1",

            "ranges": {

                "clothing": {

                    "6a00e35740a5e9c7bb2ff4dd.jpg": [
                        {
                            "start": 1,
                            "end": 3,
                            "bbox": [1, 2, 3, 4]
                        },
                        {
                            "start": 10,
                            "end": 15,
                            "bbox": [5, 6, 7, 8]
                        }
                    ]
                },

                "objects": {

                    "6a00e35740a5e9c7bb2ff4dc.jpg": [
                        {
                            "start": 5,
                            "end": 8,
                            "bbox": [10, 20, 30, 40]
                        }
                    ]
                }
            }
        },

        {
            "episodeId": "episode_2",

            "ranges": {

                "clothing": {

                    "6a00e35740a5e9c7bb2ff4de.png": [
                        {
                            "start": 2,
                            "end": 6,
                            "bbox": [100, 200, 300, 400]
                        }
                    ]
                },

                "objects": {

                    "6a00e35740a5e9c7bb2ff4dc.jpg": [
                        {
                            "start": 15,
                            "end": 20,
                            "bbox": [50, 60, 70, 80]
                        }
                    ]
                }
            }
        }
    ],

    "products": [

        {
            "productId": "6a00e35740a5e9c7bb2ff4dc",
            "imageUrl": "https://object-image.com",
            "purchaseLink": "https://object-buy.com",
            "category": "non-clothing",
            "local_image_path": (
                r"data/jobs/job1/objects/"
                r"6a00e35740a5e9c7bb2ff4dc.jpg"
            )
        },

        {
            "productId": "6a00e35740a5e9c7bb2ff4dd",
            "imageUrl": "https://cloth-a.com",
            "purchaseLink": "https://cloth-a-buy.com",
            "category": "clothing",
            "local_image_path": (
                r"data/jobs/job1/clothing/"
                r"6a00e35740a5e9c7bb2ff4dd.jpg"
            )
        },

        {
            "productId": "6a00e35740a5e9c7bb2ff4de",
            "imageUrl": "https://cloth-b.com",
            "purchaseLink": "https://cloth-b-buy.com",
            "category": "clothing",
            "local_image_path": (
                r"data/jobs/job1/clothing/"
                r"6a00e35740a5e9c7bb2ff4de.png"
            )
        }
    ]
}


# --------------------------------------------------
# TEST
# --------------------------------------------------

builder = ResponseBuilder()

response = builder.build(
    job_id=job_result["jobId"],
    series_id=job_result["seriesId"],
    episode_results=job_result["results"],
    products=job_result["products"]
)

print(
    json.dumps(
        response,
        indent=4
    )
)