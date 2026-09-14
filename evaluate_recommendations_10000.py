import csv
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION
# ============================================================

PAPERS_FILE = Path("data/processed/papers_clean_10000.csv")
EMBEDDINGS_FILE = Path("data/processed/paper_embeddings_10000.npy")

TOP_K = 5

# Papers to evaluate.
# These are spread across the 10,000-paper dataset.
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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("10,000-PAPER RECOMMENDATION EVALUATION")
print("=" * 60)

print()
print("Loading papers...")

with open(PAPERS_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

print(f"Papers loaded: {len(papers)}")

print()
print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embeddings loaded: {embeddings.shape}")


# ============================================================
# VALIDATION
# ============================================================

if len(papers) != len(embeddings):
    raise ValueError(
        f"Mismatch detected!\n"
        f"Papers: {len(papers)}\n"
        f"Embeddings: {len(embeddings)}"
    )

print()
print("Dataset and embeddings are aligned.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_categories(paper):
    """
    Convert the semicolon-separated category string
    into a set of categories.
    """

    categories = paper.get("categories", "")

    if not categories:
        return set()

    return set(
        category.strip()
        for category in categories.split(";")
        if category.strip()
    )


def calculate_category_overlap(target, candidate):
    """
    Return the shared categories between two papers.
    """

    target_categories = get_categories(target)
    candidate_categories = get_categories(candidate)

    return target_categories.intersection(candidate_categories)


def calculate_category_match(target, candidate):
    """
    Return True if the two papers share at least
    one category.
    """

    shared = calculate_category_overlap(target, candidate)

    return len(shared) > 0


# ============================================================
# EVALUATION
# ============================================================

total_comparisons = 0
category_matches = 0

all_similarity_scores = []
matched_similarity_scores = []
unmatched_similarity_scores = []

evaluation_results = []


for target_index in TARGET_INDICES:

    if target_index >= len(papers):
        print(
            f"Skipping target index {target_index}: "
            f"outside dataset."
        )
        continue

    target_paper = papers[target_index]

    print()
    print("=" * 60)
    print("TARGET PAPER")
    print("=" * 60)

    print(f"Dataset index: {target_index + 1}")
    print(f"Title: {target_paper['title']}")
    print(f"Categories: {target_paper['categories']}")

    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    target_embedding = embeddings[target_index].reshape(1, -1)

    similarities = cosine_similarity(
        target_embedding,
        embeddings
    )[0]

    # Don't recommend the target paper itself.
    similarities[target_index] = -1

    # Get top K indices.
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    print()
    print(f"TOP {TOP_K} RECOMMENDATIONS")
    print("-" * 60)

    target_comparisons = 0
    target_matches = 0

    target_scores = []

    for rank, candidate_index in enumerate(top_indices, start=1):

        candidate = papers[candidate_index]

        similarity = float(similarities[candidate_index])

        shared_categories = calculate_category_overlap(
            target_paper,
            candidate
        )

        is_match = len(shared_categories) > 0

        if is_match:
            shared_text = ", ".join(sorted(shared_categories))
            match_text = "YES"
        else:
            shared_text = "None"
            match_text = "NO"

        print()
        print(
            f"{rank}. {candidate['title']}"
        )

        print(
            f"   Similarity: {similarity:.4f}"
        )

        print(
            f"   Categories: {candidate['categories']}"
        )

        print(
            f"   Shared categories: {shared_text}"
        )

        print(
            f"   Category match: {match_text}"
        )

        # Statistics
        total_comparisons += 1
        target_comparisons += 1

        all_similarity_scores.append(similarity)
        target_scores.append(similarity)

        if is_match:
            category_matches += 1
            target_matches += 1
            matched_similarity_scores.append(similarity)
        else:
            unmatched_similarity_scores.append(similarity)

    # --------------------------------------------------------
    # Target statistics
    # --------------------------------------------------------

    target_match_rate = (
        target_matches / target_comparisons * 100
        if target_comparisons > 0
        else 0
    )

    average_target_similarity = (
        sum(target_scores) / len(target_scores)
        if target_scores
        else 0
    )

    print()
    print("-" * 60)
    print(
        f"Category match rate for this paper: "
        f"{target_match_rate:.2f}%"
    )

    print(
        f"Average similarity for this paper: "
        f"{average_target_similarity:.4f}"
    )

    evaluation_results.append({
        "target_index": target_index + 1,
        "title": target_paper["title"],
        "match_rate": target_match_rate,
        "average_similarity": average_target_similarity
    })


# ============================================================
# OVERALL RESULTS
# ============================================================

print()
print()
print("=" * 60)
print("OVERALL EVALUATION")
print("=" * 60)

overall_match_rate = (
    category_matches / total_comparisons * 100
    if total_comparisons > 0
    else 0
)

average_similarity = (
    sum(all_similarity_scores) / len(all_similarity_scores)
    if all_similarity_scores
    else 0
)

average_matched_similarity = (
    sum(matched_similarity_scores)
    / len(matched_similarity_scores)
    if matched_similarity_scores
    else 0
)

average_unmatched_similarity = (
    sum(unmatched_similarity_scores)
    / len(unmatched_similarity_scores)
    if unmatched_similarity_scores
    else 0
)

print()
print(f"Target papers evaluated: {len(evaluation_results)}")
print(f"Comparisons checked: {total_comparisons}")

print()
print(f"Category matches: {category_matches}")
print(f"Category mismatches: {total_comparisons - category_matches}")

print()
print(
    f"Overall category match rate: "
    f"{overall_match_rate:.2f}%"
)

print()
print(
    f"Average similarity: "
    f"{average_similarity:.4f}"
)

print(
    f"Average similarity of category matches: "
    f"{average_matched_similarity:.4f}"
)

print(
    f"Average similarity of category mismatches: "
    f"{average_unmatched_similarity:.4f}"
)


# ============================================================
# SUMMARY TABLE
# ============================================================

print()
print("=" * 60)
print("TARGET PAPER SUMMARY")
print("=" * 60)

for result in evaluation_results:

    print()
    print(
        f"Paper #{result['target_index']}"
    )

    print(
        f"Title: {result['title']}"
    )

    print(
        f"Category match rate: "
        f"{result['match_rate']:.2f}%"
    )

    print(
        f"Average similarity: "
        f"{result['average_similarity']:.4f}"
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

if overall_match_rate >= 80:
    print(
        "Excellent category alignment. "
        "The embedding recommendations are strongly "
        "aligned with paper categories."
    )

elif overall_match_rate >= 60:
    print(
        "Good category alignment. "
        "The embedding recommendations generally "
        "retrieve papers from related research areas."
    )

elif overall_match_rate >= 40:
    print(
        "Moderate category alignment. "
        "The embedding model is finding some related papers, "
        "but there is room for improvement."
    )

else:
    print(
        "Low category alignment. "
        "The recommendation system should be improved "
        "before being considered reliable."
    )

print()
print("=" * 60)
print("RECOMMENDATION EVALUATION COMPLETE")
print("=" * 60)