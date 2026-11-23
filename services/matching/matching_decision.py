"""Acceptance and reporting rules for product-level CLIP rankings."""
import math


CATEGORY_LABELS = {
    "matched": "Matched",
    "rejected_low_margin": "Rejected - low margin",
    "rejected_low_similarity": "Rejected - low similarity",
    "rejected_unavailable": "Rejected - comparison unavailable",
}


def decision_category(result):
    return {"accepted": "matched",
            "below_min_margin": "rejected_low_margin",
            "below_similarity_threshold": "rejected_low_similarity"}.get(
                result["decision_reason"], "rejected_unavailable")


def category_counts(rows):
    counts = dict.fromkeys(CATEGORY_LABELS, 0)
    for row in rows:
        counts[decision_category(row)] += 1
    return counts


def decide(ranking, threshold=.55, min_margin=.05):
    if not math.isfinite(min_margin) or not 0 <= min_margin <= 2:
        raise ValueError("min_margin must be finite and between 0 and 2")
    # Collapse reference rows first, so views of one product never compete.
    products = {}
    for row in ranking:
        score = float(row["similarity"])
        if not math.isfinite(score):
            raise ValueError("Non-finite similarity")
        products[row["product"]] = max(score, products.get(row["product"], -math.inf))
    ranked = sorted(products.items(), key=lambda row: row[1], reverse=True)
    best = ranked[0][1] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    margin = best - second[1] if second else None
    similarity_pass = best is not None and best >= threshold
    margin_pass = min_margin == 0 or (margin is not None and
        (margin >= min_margin or math.isclose(margin, min_margin, rel_tol=0, abs_tol=1e-7)))
    reason = ("empty_catalog" if best is None else "below_similarity_threshold" if not similarity_pass
              else "no_second_product" if min_margin > 0 and second is None
              else "below_min_margin" if not margin_pass else "accepted")
    return dict(decision_category=decision_category({"decision_reason": reason}),
                accepted=bool(similarity_pass and margin_pass),
                similarity_only_matched=similarity_pass, margin_passed=margin_pass,
                min_margin=min_margin, margin=margin, decision_reason=reason,
                second_product=second[0] if second else None,
                second_similarity=second[1] if second else None)


def comparison_counts(rows, min_margin):
    return {f"{value:g}": sum(decide(row["ranking"], row["threshold"], value)["accepted"] for row in rows)
            for value in sorted({0., .05, .10, min_margin})}
