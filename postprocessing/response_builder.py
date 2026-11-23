import os


class ResponseBuilder:

    def __init__(self):
        pass

    def _build_product_lookup(
        self,
        products
    ):

        lookup = {}
        identities = {}

        for product in products:

            filename = os.path.splitext(
                os.path.basename(
                    product["local_image_path"]
                )
            )[0]

            metadata = {

                "seriesProductId":
                    product["productId"],

                "purchaseLink":
                    product["purchaseLink"],

                "imageUrl":
                    product["imageUrl"],

                "imageKey":
                    product["imageKey"],

                "category":
                    product["category"]
            }
            identities[product["productId"]] = metadata
            lookup[filename] = metadata
            lookup[os.path.basename(product["local_image_path"])] = metadata

        # Explicit IDs win over coincidentally identical legacy filename aliases.
        lookup.update(identities)
        return lookup

    def build(
        self,
        job_id,
        series_id,
        episode_results,
        products
    ):

        product_lookup = (
            self._build_product_lookup(
                products
            )
        )

        response = {

            "jobId": job_id,

            "seriesId": series_id,

            "status": "completed",

            "episodes": []
        }

        for episode in episode_results:

            episode_payload = {

                "episodeId":
                    episode["episodeId"],

                "clothing": [],

                "objects": []
            }

            ranges = episode["ranges"]

            # ----------------------------------
            # CLOTHING
            # ----------------------------------
            # Clothing is category-based
            # Example:
            # shirt -> [...]
            # trousers -> [...]
            # dress -> [...]
            # ----------------------------------

            for (
                clothing_type,
                clothing_ranges
            ) in ranges.get(
                "clothing",
                {}
            ).items():

                episode_payload[
                    "clothing"
                ].append(
                    {
                        "category":
                            clothing_type,

                        "ranges":
                            clothing_ranges
                    }
                )

            # ----------------------------------
            # OBJECTS
            # ----------------------------------
            # Objects are product-mapped
            # Example:
            # 6a00e35740a5e9c7bb2ff4dc.jpg -> [...]
            # ----------------------------------

            for (
                filename,
                product_ranges
            ) in ranges.get(
                "objects",
                {}
            ).items():
                metadata = (
                    product_lookup.get(
                        filename
                    )
                )

                if not metadata:
                    raise ValueError(f"Matched product has no request metadata: {filename}")

                episode_payload[
                    "objects"
                ].append(
                    {

                        "seriesProductId":
                            metadata[
                                "seriesProductId"
                            ],

                        "category":
                            metadata[
                                "category"
                            ],

                        "imageKey":
                            metadata[
                                "imageKey"
                            ],

                        "purchaseLink":
                            metadata[
                                "purchaseLink"
                            ],

                        "imageUrl":
                            metadata[
                                "imageUrl"
                            ],

                        "ranges":
                            product_ranges
                    }
                )

            response[
                "episodes"
            ].append(
                episode_payload
            )

        return response
