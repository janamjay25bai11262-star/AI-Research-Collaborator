import csv
import numpy as np
from pathlib import Path

# ============================================================
# GENERAL CATEGORY TOP-K EVALUATION
# ALL-CATEGORY MATCHING VERSION
# ============================================================

PAPER_FILE = Path("data/processed/papers_clean_10000.csv")
EMBEDDING_FILE = Path("data/processed/paper_embeddings_10000.npy")

TOP_KS = [1, 3, 5, 10, 20]
MAX_TARGETS = 10


def parse_categories(value):
    """Convert 'cs.RO;cs.AI;cs.LG' into a set."""
    if not value:
        return set()

    return {
        x.strip()
        for x in value.replace(",", ";").split(";")
        if x.strip()
    }


def normalize_embeddings(embeddings):
    """Normalize vectors so dot product gives cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def all_categories_match(requested, paper_categories):
    """
    True ONLY when the paper contains EVERY category requested.

    Example:
      requested = {cs.RO, cs.AI, cs.LG}

      cs.RO;cs.AI;cs.LG       -> MATCH
      cs.RO;cs.AI             -> NO MATCH
      cs.AI;cs.LG             -> NO MATCH
      cs.RO;cs.AI;cs.LG;cs.CV -> MATCH
    """
    return requested.issubset(paper_categories)


print("=" * 70)
print("GENERAL CATEGORY TOP-K EVALUATION - ALL-CATEGORY MATCHING")
print("=" * 70)

# ------------------------------------------------------------
# Load papers
# ------------------------------------------------------------

if not PAPER_FILE.exists():
    raise FileNotFoundError(
        f"\nPaper file not found:\n{PAPER_FILE}\n"
        "Check the file path in PAPER_FILE."
    )

with open(PAPER_FILE, "r", encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

print(f"\nPapers loaded: {len(papers)}")

# ------------------------------------------------------------
# Load embeddings
# ------------------------------------------------------------

if not EMBEDDING_FILE.exists():
    raise FileNotFoundError(
        f"\nEmbedding file not found:\n{EMBEDDING_FILE}\n"
        "Check the file path in EMBEDDING_FILE."
    )

embeddings = np.load(EMBEDDING_FILE)

print(f"Embeddings loaded: {embeddings.shape}")

if len(papers) != embeddings.shape[0]:
    raise ValueError(
        f"Dataset/embedding mismatch: {len(papers)} papers vs "
        f"{embeddings.shape[0]} embeddings."
    )

embeddings = normalize_embeddings(embeddings)
print("Embeddings normalized.")

# ------------------------------------------------------------
# Ask for category/categories
# ------------------------------------------------------------

raw = input(
    "\nEnter category/categories "
    "(example: cs.RO or cs.RO,cs.AI,cs.LG): "
).strip()

requested_categories = parse_categories(raw)

if not requested_categories:
    raise ValueError("No valid category was entered.")

print(
    "\nRequested categories:",
    ", ".join(sorted(requested_categories))
)

# ------------------------------------------------------------
# Find target papers that contain ALL requested categories
# ------------------------------------------------------------

target_indices = []

for i, paper in enumerate(papers):
    paper_categories = parse_categories(paper.get("categories", ""))

    if all_categories_match(requested_categories, paper_categories):
        target_indices.append(i)

print(
    f"Papers containing ALL requested categories: "
    f"{len(target_indices)}"
)

if not target_indices:
    print(
        "\nNo papers contain ALL of the requested categories.\n"
        "Try fewer categories or categories that commonly occur together."
    )
    raise SystemExit

# Select up to MAX_TARGETS representative targets
if len(target_indices) > MAX_TARGETS:
    positions = np.linspace(
        0, len(target_indices) - 1, MAX_TARGETS, dtype=int
    )
    sample_indices = [target_indices[p] for p in positions]
else:
    sample_indices = target_indices

print(f"Target papers evaluated: {len(sample_indices)}")

# ------------------------------------------------------------
# Overall statistics
# ------------------------------------------------------------

overall = {
    k: {
        "targets_with_match": 0,
        "matching_recommendations": 0,
        "total_recommendations": 0,
    }
    for k in TOP_KS
}

# ============================================================
# EVALUATION
# ============================================================

for target_index in sample_indices:

    target = papers[target_index]

    scores = embeddings @ embeddings[target_index]

    # Do not recommend the target paper itself
    scores[target_index] = -1.0

    ranked = np.argsort(scores)[::-1]

    print("\n" + "=" * 70)
    print("TARGET PAPER")
    print("=" * 70)
    print(f"Dataset position: {target_index + 1}")
    print(f"Title: {target.get('title', '')}")
    print(f"All paper categories: {target.get('categories', '')}")
    print(
        "Required categories: "
        + ", ".join(sorted(requested_categories))
    )

    for k in TOP_KS:

        top_indices = ranked[:k]

        matching_recommendations = 0

        for idx in top_indices:

            recommended_categories = parse_categories(
                papers[idx].get("categories", "")
            )

            if all_categories_match(
                requested_categories,
                recommended_categories
            ):
                matching_recommendations += 1

        if matching_recommendations > 0:
            overall[k]["targets_with_match"] += 1

        overall[k]["matching_recommendations"] += matching_recommendations
        overall[k]["total_recommendations"] += k

        coverage = (
            100 * matching_recommendations / k
            if k else 0
        )

        print(
            f"\nTop-{k}: "
            f"{matching_recommendations}/{k} recommendations contain "
            f"ALL requested categories | "
            f"Target success: "
            f"{'YES' if matching_recommendations > 0 else 'NO'} | "
            f"All-category match rate: {coverage:.2f}%"
        )

# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL ALL-CATEGORY EVALUATION")
print("=" * 70)

number_of_targets = len(sample_indices)

for k in TOP_KS:

    success_rate = (
        100 * overall[k]["targets_with_match"] / number_of_targets
        if number_of_targets else 0
    )

    match_rate = (
        100 * overall[k]["matching_recommendations"]
        / overall[k]["total_recommendations"]
        if overall[k]["total_recommendations"] else 0
    )

    print(
        f"Top-{k}: "
        f"{overall[k]['targets_with_match']}/{number_of_targets} "
        f"target papers had at least one recommendation containing "
        f"ALL requested categories "
        f"({success_rate:.2f}%) | "
        f"All-category match rate: {match_rate:.2f}%"
    )

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nMatching rule: a recommendation is a MATCH only if it contains "
    "EVERY category entered by the user."
)

print(
    "Note: This is a category-overlap evaluation. It measures exact "
    "category-set inclusion and is not a human relevance judgment."
)
