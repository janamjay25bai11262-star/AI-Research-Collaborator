import csv
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PAPERS_FILE = Path("data/processed/papers_clean_10000.csv")
EMBEDDINGS_FILE = Path("data/processed/paper_embeddings_10000.npy")

# K values to evaluate
K_VALUES = [1, 3, 5, 10, 20]

# Papers to use as evaluation targets
TARGET_INDICES = [
    0,
    999,
    1999,
    2999,
    3999,
    4999,
    5999,
    6999,
    7999,
    8999,
    9999
]

# arXiv broad categories that are excluded when checking
# "specific-category" overlap.
BROAD_CATEGORIES = {
    "cs.AI"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_categories(category_string):
    """
    Convert:
        cs.AI;cs.LG;cs.CL

    into:
        {"cs.AI", "cs.LG", "cs.CL"}
    """
    if not category_string:
        return set()

    return {
        category.strip()
        for category in category_string.split(";")
        if category.strip()
    }


def get_specific_categories(category_string):
    """
    Remove broad categories such as cs.AI.

    Example:

        cs.CL;cs.AI;cs.LG

    becomes:

        {"cs.CL", "cs.LG"}
    """
    categories = parse_categories(category_string)

    return categories - BROAD_CATEGORIES


def cosine_similarity_matrix(embeddings):
    """
    Normalize embeddings so dot product becomes cosine similarity.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Prevent division by zero
    norms[norms == 0] = 1.0

    return embeddings / norms


def get_top_k_recommendations(
    normalized_embeddings,
    target_index,
    k
):
    """
    Return the top-k papers most similar to the target paper.

    The target paper itself is excluded.
    """

    target_embedding = normalized_embeddings[target_index]

    similarities = np.dot(
        normalized_embeddings,
        target_embedding
    )

    # Exclude the target paper itself
    similarities[target_index] = -1.0

    # Get highest similarity values
    top_indices = np.argsort(similarities)[::-1][:k]

    return [
        (int(index), float(similarities[index]))
        for index in top_indices
    ]


def has_specific_category_match(
    target_specific_categories,
    recommendation_specific_categories
):
    """
    Check whether target and recommendation share
    at least one specific category.
    """

    return bool(
        target_specific_categories
        & recommendation_specific_categories
    )


# ============================================================
# LOAD PAPERS
# ============================================================

print("=" * 60)
print("10,000-PAPER TOP-K RECOMMENDATION EVALUATION")
print("=" * 60)

print()
print("Loading papers...")

with open(
    PAPERS_FILE,
    "r",
    encoding="utf-8"
) as file:
    papers = list(csv.DictReader(file))

print(f"Papers loaded: {len(papers)}")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print()
print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embeddings loaded: {embeddings.shape}")


# ============================================================
# ALIGNMENT CHECK
# ============================================================

if len(papers) != embeddings.shape[0]:
    raise ValueError(
        "ERROR: Number of papers does not match "
        "number of embeddings."
    )

print()
print("Dataset and embeddings are aligned.")


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

print()
print("Normalizing embeddings...")

normalized_embeddings = cosine_similarity_matrix(
    embeddings
)

print("Embeddings normalized.")


# ============================================================
# VALIDATE TARGET INDICES
# ============================================================

for index in TARGET_INDICES:
    if index < 0 or index >= len(papers):
        raise IndexError(
            f"Target index {index} is outside the dataset."
        )


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

results = {
    k: {
        "targets_with_match": 0,
        "targets_without_match": 0,
        "total_matches": 0,
        "total_recommendations": 0,
        "average_similarity": []
    }
    for k in K_VALUES
}


# ============================================================
# EVALUATION
# ============================================================

for target_index in TARGET_INDICES:

    target = papers[target_index]

    target_categories = parse_categories(
        target["categories"]
    )

    target_specific_categories = (
        get_specific_categories(
            target["categories"]
        )
    )

    print()
    print("=" * 60)
    print("TARGET PAPER")
    print("=" * 60)

    print(f"Dataset index: {target_index + 1}")
    print(f"Title: {target['title']}")
    print(f"All categories: {target['categories']}")

    if target_specific_categories:
        print(
            "Specific categories: "
            + ", ".join(sorted(target_specific_categories))
        )
    else:
        print("Specific categories: None")

    # --------------------------------------------------------
    # Calculate all similarities once for this target
    # --------------------------------------------------------

    target_embedding = normalized_embeddings[target_index]

    similarities = np.dot(
        normalized_embeddings,
        target_embedding
    )

    similarities[target_index] = -1.0

    sorted_indices = np.argsort(
        similarities
    )[::-1]

    # --------------------------------------------------------
    # Evaluate each K
    # --------------------------------------------------------

    for k in K_VALUES:

        top_indices = sorted_indices[:k]

        matches = 0
        similarity_values = []

        for recommendation_index in top_indices:

            similarity = float(
                similarities[recommendation_index]
            )

            similarity_values.append(similarity)

            recommendation = papers[
                recommendation_index
            ]

            recommendation_specific_categories = (
                get_specific_categories(
                    recommendation["categories"]
                )
            )

            if has_specific_category_match(
                target_specific_categories,
                recommendation_specific_categories
            ):
                matches += 1

        # ----------------------------------------------------
        # Update aggregate statistics
        # ----------------------------------------------------

        results[k]["total_matches"] += matches
        results[k]["total_recommendations"] += k

        results[k]["average_similarity"].extend(
            similarity_values
        )

        # Top-K success:
        # Did at least ONE of the top-K recommendations
        # share a specific category?
        if matches > 0:
            results[k]["targets_with_match"] += 1
        else:
            results[k]["targets_without_match"] += 1


# ============================================================
# OVERALL RESULTS
# ============================================================

print()
print("=" * 60)
print("OVERALL TOP-K EVALUATION")
print("=" * 60)

number_of_targets = len(TARGET_INDICES)

print()
print(f"Target papers evaluated: {number_of_targets}")
print(f"K values evaluated: {K_VALUES}")

print()
print("-" * 60)
print("TOP-K RESULTS")
print("-" * 60)

for k in K_VALUES:

    targets_with_match = results[k][
        "targets_with_match"
    ]

    targets_without_match = results[k][
        "targets_without_match"
    ]

    total_matches = results[k][
        "total_matches"
    ]

    total_recommendations = results[k][
        "total_recommendations"
    ]

    average_similarity = np.mean(
        results[k]["average_similarity"]
    )

    # Recall-style Top-K success rate
    top_k_success_rate = (
        targets_with_match
        / number_of_targets
        * 100
    )

    # Percentage of all recommendations that
    # shared a specific category
    category_match_rate = (
        total_matches
        / total_recommendations
        * 100
    )

    print()
    print(f"TOP-{k}")
    print("-" * 40)

    print(
        f"Targets with >=1 category match: "
        f"{targets_with_match}/{number_of_targets}"
    )

    print(
        f"Targets with no category match: "
        f"{targets_without_match}/{number_of_targets}"
    )

    print(
        f"Top-{k} success rate: "
        f"{top_k_success_rate:.2f}%"
    )

    print(
        f"Individual recommendation category "
        f"match rate: {category_match_rate:.2f}%"
    )

    print(
        f"Average similarity: "
        f"{average_similarity:.4f}"
    )


# ============================================================
# TARGET-BY-TARGET SUMMARY
# ============================================================

print()
print("=" * 60)
print("TARGET-BY-TARGET TOP-K SUMMARY")
print("=" * 60)

for target_index in TARGET_INDICES:

    target = papers[target_index]

    target_specific_categories = (
        get_specific_categories(
            target["categories"]
        )
    )

    target_embedding = normalized_embeddings[
        target_index
    ]

    similarities = np.dot(
        normalized_embeddings,
        target_embedding
    )

    similarities[target_index] = -1.0

    sorted_indices = np.argsort(
        similarities
    )[::-1]

    print()
    print(
        f"Paper #{target_index + 1}: "
        f"{target['title']}"
    )

    if target_specific_categories:
        print(
            "Specific categories: "
            + ", ".join(
                sorted(target_specific_categories)
            )
        )
    else:
        print(
            "Specific categories: None"
        )

    for k in K_VALUES:

        top_indices = sorted_indices[:k]

        match_count = 0

        for recommendation_index in top_indices:

            recommendation = papers[
                recommendation_index
            ]

            recommendation_specific_categories = (
                get_specific_categories(
                    recommendation["categories"]
                )
            )

            if has_specific_category_match(
                target_specific_categories,
                recommendation_specific_categories
            ):
                match_count += 1

        success = match_count > 0

        print(
            f"  Top-{k}: "
            f"{match_count}/{k} specific-category matches "
            f"| Success: {'YES' if success else 'NO'}"
        )


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

top_1_success = (
    results[1]["targets_with_match"]
    / number_of_targets
    * 100
)

top_5_success = (
    results[5]["targets_with_match"]
    / number_of_targets
    * 100
)

top_10_success = (
    results[10]["targets_with_match"]
    / number_of_targets
    * 100
)

top_20_success = (
    results[20]["targets_with_match"]
    / number_of_targets
    * 100
)

print()

print(
    f"Top-1 success rate:  {top_1_success:.2f}%"
)

print(
    f"Top-5 success rate:  {top_5_success:.2f}%"
)

print(
    f"Top-10 success rate: {top_10_success:.2f}%"
)

print(
    f"Top-20 success rate: {top_20_success:.2f}%"
)

print()

if top_10_success >= 80:
    print(
        "Strong Top-K category retrieval performance."
    )
elif top_10_success >= 50:
    print(
        "Moderate Top-K category retrieval performance."
    )
else:
    print(
        "Weak Top-K category retrieval performance."
    )

print()
print("=" * 60)
print("TOP-K EVALUATION COMPLETE")
print("=" * 60)