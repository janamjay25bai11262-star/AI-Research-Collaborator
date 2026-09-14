import csv
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PAPERS_FILE = Path("data/processed/papers_clean_10000.csv")
EMBEDDINGS_FILE = Path("data/processed/paper_embeddings_10000.npy")

TOP_K = 5

# Broad category that appears on essentially every paper
BROAD_CATEGORY = "cs.AI"

# Evaluate these dataset positions
TARGET_INDICES = [
    0,       # Paper #1
    999,     # Paper #1000
    1999,    # Paper #2000
    2999,    # Paper #3000
    3999,    # Paper #4000
    4999,    # Paper #5000
    5999,    # Paper #6000
    6999,    # Paper #7000
    7999,    # Paper #8000
    8999,    # Paper #9000
    9999     # Paper #10000
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_categories(category_string):
    """
    Convert:
        cs.LG;cs.AI;stat.ML

    into:
        {"cs.LG", "cs.AI", "stat.ML"}
    """
    if not category_string:
        return set()

    return {
        category.strip()
        for category in category_string.split(";")
        if category.strip()
    }


def specific_categories(category_string):
    """
    Remove broad cs.AI category.

    Example:
        cs.LG;cs.AI;stat.ML
    becomes:
        {"cs.LG", "stat.ML"}
    """
    categories = parse_categories(category_string)
    categories.discard(BROAD_CATEGORY)
    return categories


def cosine_similarity_matrix(embeddings):
    """
    Normalize embeddings so dot product becomes cosine similarity.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Prevent division by zero
    norms[norms == 0] = 1.0

    return embeddings / norms


# ============================================================
# START
# ============================================================

print("=" * 60)
print("10,000-PAPER SPECIFIC CATEGORY EVALUATION")
print("=" * 60)


# ============================================================
# LOAD PAPERS
# ============================================================

print("\nLoading papers...")

if not PAPERS_FILE.exists():
    raise FileNotFoundError(
        f"Paper dataset not found:\n{PAPERS_FILE}"
    )

with open(PAPERS_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

print(f"Papers loaded: {len(papers)}")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

if not EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(
        f"Embedding file not found:\n{EMBEDDINGS_FILE}"
    )

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embeddings loaded: {embeddings.shape}")


# ============================================================
# ALIGNMENT CHECK
# ============================================================

if len(papers) != embeddings.shape[0]:
    raise ValueError(
        "\nERROR: Dataset and embeddings are not aligned!\n"
        f"Papers: {len(papers)}\n"
        f"Embeddings: {embeddings.shape[0]}"
    )

print("\nDataset and embeddings are aligned.")


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

print("\nNormalizing embeddings...")

normalized_embeddings = cosine_similarity_matrix(embeddings)


# ============================================================
# EVALUATION VARIABLES
# ============================================================

total_comparisons = 0
total_specific_matches = 0

all_similarities = []
matched_similarities = []
mismatched_similarities = []

paper_results = []


# ============================================================
# EVALUATE TARGET PAPERS
# ============================================================

for target_index in TARGET_INDICES:

    if target_index >= len(papers):
        print(
            f"\nWARNING: Target index {target_index + 1} "
            f"is outside dataset."
        )
        continue

    target = papers[target_index]

    target_categories = specific_categories(
        target["categories"]
    )

    print("\n" + "=" * 60)
    print("TARGET PAPER")
    print("=" * 60)

    print(f"Dataset index: {target_index + 1}")

    print(f"Title: {target['title']}")

    print(
        "All categories: "
        f"{target['categories']}"
    )

    if target_categories:
        print(
            "Specific categories: "
            f"{', '.join(sorted(target_categories))}"
        )
    else:
        print("Specific categories: None")

    # --------------------------------------------------------
    # Calculate similarity against every paper
    # --------------------------------------------------------

    target_embedding = normalized_embeddings[target_index]

    similarities = normalized_embeddings @ target_embedding

    # Exclude target itself
    similarities[target_index] = -1

    # Get top K
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nTOP 5 RECOMMENDATIONS")
    print("-" * 60)

    paper_matches = 0
    paper_similarities = []

    for rank, index in enumerate(top_indices, start=1):

        recommended = papers[index]

        similarity = float(similarities[index])

        recommended_categories = specific_categories(
            recommended["categories"]
        )

        shared_categories = (
            target_categories & recommended_categories
        )

        category_match = len(shared_categories) > 0

        total_comparisons += 1

        all_similarities.append(similarity)
        paper_similarities.append(similarity)

        if category_match:
            total_specific_matches += 1
            paper_matches += 1
            matched_similarities.append(similarity)
        else:
            mismatched_similarities.append(similarity)

        print(f"\n{rank}. {recommended['title']}")

        print(
            f"   Similarity: {similarity:.4f}"
        )

        print(
            f"   All categories: "
            f"{recommended['categories']}"
        )

        if recommended_categories:
            print(
                "   Specific categories: "
                f"{', '.join(sorted(recommended_categories))}"
            )
        else:
            print(
                "   Specific categories: None"
            )

        if shared_categories:
            print(
                "   Shared specific categories: "
                f"{', '.join(sorted(shared_categories))}"
            )
        else:
            print(
                "   Shared specific categories: None"
            )

        print(
            "   Specific-category match: "
            f"{'YES' if category_match else 'NO'}"
        )

    # --------------------------------------------------------
    # Per-paper statistics
    # --------------------------------------------------------

    match_rate = (
        paper_matches / TOP_K * 100
    )

    average_similarity = (
        sum(paper_similarities)
        / len(paper_similarities)
    )

    print("\n" + "-" * 60)

    print(
        "Specific-category match rate for this paper: "
        f"{match_rate:.2f}%"
    )

    print(
        "Average similarity for this paper: "
        f"{average_similarity:.4f}"
    )

    paper_results.append({
        "index": target_index + 1,
        "title": target["title"],
        "match_rate": match_rate,
        "average_similarity": average_similarity
    })


# ============================================================
# OVERALL EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("OVERALL EVALUATION")
print("=" * 60)

print(
    f"\nTarget papers evaluated: "
    f"{len(paper_results)}"
)

print(
    f"Comparisons checked: "
    f"{total_comparisons}"
)

print(
    f"Specific-category matches: "
    f"{total_specific_matches}"
)

print(
    f"Specific-category mismatches: "
    f"{total_comparisons - total_specific_matches}"
)


if total_comparisons > 0:

    overall_match_rate = (
        total_specific_matches
        / total_comparisons
        * 100
    )

    overall_average_similarity = (
        sum(all_similarities)
        / len(all_similarities)
    )

    print(
        f"\nOverall specific-category match rate: "
        f"{overall_match_rate:.2f}%"
    )

    print(
        f"Overall average similarity: "
        f"{overall_average_similarity:.4f}"
    )

else:

    overall_match_rate = 0
    overall_average_similarity = 0

    print(
        "\nNo comparisons were performed."
    )


# ============================================================
# MATCHED / MISMATCHED SIMILARITY
# ============================================================

print("\n" + "=" * 60)
print("SIMILARITY BREAKDOWN")
print("=" * 60)

if matched_similarities:

    avg_matched = (
        sum(matched_similarities)
        / len(matched_similarities)
    )

    print(
        f"\nAverage similarity of "
        f"specific-category matches: "
        f"{avg_matched:.4f}"
    )
else:

    print(
        "\nAverage similarity of "
        "specific-category matches: N/A"
    )


if mismatched_similarities:

    avg_mismatched = (
        sum(mismatched_similarities)
        / len(mismatched_similarities)
    )

    print(
        f"Average similarity of "
        f"specific-category mismatches: "
        f"{avg_mismatched:.4f}"
    )
else:

    print(
        "Average similarity of "
        "specific-category mismatches: N/A"
    )


# ============================================================
# TARGET SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TARGET PAPER SUMMARY")
print("=" * 60)

for result in paper_results:

    print(
        f"\nPaper #{result['index']}"
    )

    print(
        f"Title: {result['title']}"
    )

    print(
        "Specific-category match rate: "
        f"{result['match_rate']:.2f}%"
    )

    print(
        "Average similarity: "
        f"{result['average_similarity']:.4f}"
    )


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

if overall_match_rate >= 80:

    print(
        "\nStrong specific-category alignment."
    )

elif overall_match_rate >= 60:

    print(
        "\nGood specific-category alignment, "
        "but there is room for improvement."
    )

elif overall_match_rate >= 40:

    print(
        "\nModerate specific-category alignment."
    )

else:

    print(
        "\nWeak specific-category alignment. "
        "The recommendation system may need improvement."
    )


print("\n" + "=" * 60)
print("SPECIFIC CATEGORY EVALUATION COMPLETE")
print("=" * 60)