import csv
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROFILE_FILE = Path(
    "data/processed/researcher_profiles_clean.csv"
)

RESEARCHER_EMBEDDING_FILE = Path(
    "data/processed/researcher_embeddings.npy"
)

PAPER_FILE = Path(
    "data/processed/papers_clean_10000.csv"
)

PAPER_EMBEDDING_FILE = Path(
    "data/processed/paper_embeddings_10000.npy"
)

K_VALUES = [1, 5, 10, 20]

# Number of researchers to evaluate.
# Set to None to evaluate ALL researchers.
MAX_RESEARCHERS = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_categories(category_string):
    """
    Convert a semicolon-separated category string
    into a set.

    Example:
        'cs.AI;cs.LG;cs.CL'

    becomes:
        {'cs.AI', 'cs.LG', 'cs.CL'}
    """

    if not category_string:
        return set()

    return {
        category.strip()
        for category in category_string.split(";")
        if category.strip()
    }


def normalize_embeddings(embeddings):
    """
    L2-normalize embeddings so that dot product
    becomes cosine similarity.
    """

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    # Prevent division by zero
    norms[norms == 0] = 1.0

    return embeddings / norms


def get_top_k_indices(scores, k):
    """
    Return indices of the top-k papers.
    """

    # argpartition is much faster than sorting all 10,000
    # papers when only the top-k are required.
    top_indices = np.argpartition(
        scores,
        -k
    )[-k:]

    # Sort the selected indices by similarity
    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    return top_indices


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("RESEARCHER -> 10,000-PAPER RECOMMENDATION EVALUATION")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    PROFILE_FILE,
    RESEARCHER_EMBEDDING_FILE,
    PAPER_FILE,
    PAPER_EMBEDDING_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        print("\nERROR: Required file not found:")
        print(file_path)

        raise FileNotFoundError(file_path)


# ============================================================
# LOAD RESEARCHER PROFILES
# ============================================================

print("\nLoading researcher profiles...")

with open(
    PROFILE_FILE,
    "r",
    encoding="utf-8"
) as file:

    researchers = list(
        csv.DictReader(file)
    )

print(
    f"Researchers found: {len(researchers)}"
)


# ============================================================
# LOAD PAPERS
# ============================================================

print("\nLoading papers...")

with open(
    PAPER_FILE,
    "r",
    encoding="utf-8"
) as file:

    papers = list(
        csv.DictReader(file)
    )

print(
    f"Papers found: {len(papers)}"
)


# ============================================================
# LOAD RESEARCHER EMBEDDINGS
# ============================================================

print("\nLoading researcher embeddings...")

researcher_embeddings = np.load(
    RESEARCHER_EMBEDDING_FILE
)

print(
    "Researcher embeddings:",
    researcher_embeddings.shape
)


# ============================================================
# LOAD PAPER EMBEDDINGS
# ============================================================

print("\nLoading paper embeddings...")

paper_embeddings = np.load(
    PAPER_EMBEDDING_FILE
)

print(
    "Paper embeddings:",
    paper_embeddings.shape
)


# ============================================================
# VALIDATION
# ============================================================

if len(researchers) != researcher_embeddings.shape[0]:

    raise ValueError(
        "Researcher profile count does not match "
        "researcher embedding count."
    )


if len(papers) != paper_embeddings.shape[0]:

    raise ValueError(
        "Paper count does not match paper embedding count."
    )


if researcher_embeddings.shape[1] != paper_embeddings.shape[1]:

    raise ValueError(
        "Researcher and paper embedding dimensions do not match."
    )


print("\nDataset and embeddings are aligned.")


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

print("\nNormalizing embeddings...")

researcher_embeddings = normalize_embeddings(
    researcher_embeddings
)

paper_embeddings = normalize_embeddings(
    paper_embeddings
)

print("Embeddings normalized.")


# ============================================================
# SELECT RESEARCHERS
# ============================================================

if MAX_RESEARCHERS is None:

    researcher_indices = range(
        len(researchers)
    )

else:

    researcher_indices = range(
        min(
            MAX_RESEARCHERS,
            len(researchers)
        )
    )


researcher_indices = list(
    researcher_indices
)


# ============================================================
# STORAGE FOR OVERALL RESULTS
# ============================================================

overall_results = {}

for k in K_VALUES:

    overall_results[k] = {
        "researchers_with_match": 0,
        "researchers_without_match": 0,
        "recommendations": 0,
        "category_matches": 0,
        "similarity_sum": 0.0,
    }


# ============================================================
# TARGET-BY-TARGET RESULTS
# ============================================================

target_results = []


# ============================================================
# EVALUATION
# ============================================================

for researcher_index in researcher_indices:

    researcher = researchers[researcher_index]

    researcher_name = researcher.get(
        "researcher_name",
        f"Researcher #{researcher_index + 1}"
    )

    research_text = researcher.get(
        "research_text",
        ""
    )

    print("\n" + "=" * 60)
    print("TARGET RESEARCHER")
    print("=" * 60)

    print(
        f"Researcher index: {researcher_index + 1}"
    )

    print(
        f"Name: {researcher_name}"
    )

    if research_text:

        preview = research_text[:250]

        if len(research_text) > 250:
            preview += "..."

        print(
            f"Research text: {preview}"
        )

    # --------------------------------------------------------
    # Calculate researcher -> all paper similarities
    # --------------------------------------------------------

    researcher_vector = (
        researcher_embeddings[researcher_index]
    )

    scores = np.dot(
        paper_embeddings,
        researcher_vector
    )

    # --------------------------------------------------------
    # Evaluate each K
    # --------------------------------------------------------

    researcher_result = {
        "researcher_index": researcher_index + 1,
        "researcher_name": researcher_name,
        "results": {}
    }

    for k in K_VALUES:

        top_indices = get_top_k_indices(
            scores,
            k
        )

        print("\n" + "-" * 60)
        print(f"TOP-{k} PAPER RECOMMENDATIONS")
        print("-" * 60)

        category_match_count = 0
        similarity_sum = 0.0

        # ----------------------------------------------------
        # Display recommendations
        # ----------------------------------------------------

        for rank, paper_index in enumerate(
            top_indices,
            start=1
        ):

            paper = papers[paper_index]

            title = paper.get(
                "title",
                "Unknown title"
            )

            similarity = float(
                scores[paper_index]
            )

            categories = parse_categories(
                paper.get(
                    "categories",
                    ""
                )
            )

            print(
                f"\n{rank}. {title}"
            )

            print(
                f"   Similarity: {similarity:.4f}"
            )

            print(
                f"   Categories: "
                f"{paper.get('categories', 'None')}"
            )

            similarity_sum += similarity

            # ------------------------------------------------
            # Category evaluation
            #
            # A researcher may not have categories, because
            # researcher_profiles_clean.csv may only contain
            # research_text.
            #
            # If researcher categories exist, compare them.
            # ------------------------------------------------

            researcher_category_string = (
                researcher.get(
                    "categories",
                    researcher.get(
                        "category",
                        ""
                    )
                )
            )

            researcher_categories = parse_categories(
                researcher_category_string
            )

            shared_categories = (
                researcher_categories
                & categories
            )

            if researcher_categories:

                if shared_categories:

                    category_match_count += 1

                    print(
                        "   Shared categories: "
                        + ", ".join(
                            sorted(shared_categories)
                        )
                    )

                    print(
                        "   Category match: YES"
                    )

                else:

                    print(
                        "   Shared categories: None"
                    )

                    print(
                        "   Category match: NO"
                    )

            else:

                print(
                    "   Category evaluation: "
                    "N/A (no researcher categories)"
                )

        # ----------------------------------------------------
        # Calculate statistics for this K
        # ----------------------------------------------------

        average_similarity = (
            similarity_sum / k
        )

        researcher_has_category_match = (
            category_match_count > 0
        )

        if researcher_categories:

            if researcher_has_category_match:

                overall_results[k][
                    "researchers_with_match"
                ] += 1

            else:

                overall_results[k][
                    "researchers_without_match"
                ] += 1

            overall_results[k][
                "category_matches"
            ] += category_match_count

            overall_results[k][
                "recommendations"
            ] += k

        overall_results[k][
            "similarity_sum"
        ] += average_similarity

        researcher_result["results"][k] = {
            "category_matches": category_match_count,
            "category_match_rate": (
                category_match_count / k
                if researcher_categories
                else None
            ),
            "success": (
                researcher_has_category_match
                if researcher_categories
                else None
            ),
            "average_similarity": average_similarity,
        }

        print(
            f"\nTop-{k} average similarity: "
            f"{average_similarity:.4f}"
        )

        if researcher_categories:

            print(
                f"Top-{k} category matches: "
                f"{category_match_count}/{k}"
            )

            print(
                f"Top-{k} category match rate: "
                f"{(category_match_count / k) * 100:.2f}%"
            )

            print(
                "Top-{0} success: {1}".format(
                    k,
                    "YES"
                    if researcher_has_category_match
                    else "NO"
                )
            )

    target_results.append(
        researcher_result
    )


# ============================================================
# OVERALL EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("OVERALL RESEARCHER -> PAPER TOP-K EVALUATION")
print("=" * 60)

print(
    f"\nResearchers evaluated: "
    f"{len(researcher_indices)}"
)

print(
    f"Papers available per researcher: "
    f"{len(papers)}"
)

print(
    f"K values evaluated: "
    f"{K_VALUES}"
)


# ============================================================
# CHECK WHETHER CATEGORY EVALUATION IS POSSIBLE
# ============================================================

researchers_with_categories = 0

for researcher in researchers:

    categories = researcher.get(
        "categories",
        researcher.get(
            "category",
            ""
        )
    )

    if parse_categories(categories):

        researchers_with_categories += 1


print(
    f"Researchers with category information: "
    f"{researchers_with_categories}"
)


# ============================================================
# PRINT TOP-K RESULTS
# ============================================================

for k in K_VALUES:

    result = overall_results[k]

    print("\n" + "-" * 60)
    print(f"TOP-{k}")
    print("-" * 60)

    average_similarity = (
        result["similarity_sum"]
        / len(researcher_indices)
    )

    print(
        f"Average similarity: "
        f"{average_similarity:.4f}"
    )

    if researchers_with_categories > 0:

        success_rate = (
            result["researchers_with_match"]
            / researchers_with_categories
        )

        individual_match_rate = (
            result["category_matches"]
            / result["recommendations"]
            if result["recommendations"] > 0
            else 0
        )

        print(
            "Researchers with >=1 category match: "
            f"{result['researchers_with_match']}/"
            f"{researchers_with_categories}"
        )

        print(
            "Researchers with no category match: "
            f"{result['researchers_without_match']}/"
            f"{researchers_with_categories}"
        )

        print(
            f"Top-{k} success rate: "
            f"{success_rate * 100:.2f}%"
        )

        print(
            "Individual recommendation category "
            f"match rate: "
            f"{individual_match_rate * 100:.2f}%"
        )

    else:

        print(
            "Category evaluation: NOT AVAILABLE"
        )

        print(
            "Reason: researcher profiles do not "
            "contain category information."
        )


# ============================================================
# TARGET SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("RESEARCHER-BY-RESEARCHER TOP-K SUMMARY")
print("=" * 60)


for result in target_results:

    print(
        f"\nResearcher #{result['researcher_index']}: "
        f"{result['researcher_name']}"
    )

    for k in K_VALUES:

        data = result["results"][k]

        if data["category_match_rate"] is not None:

            print(
                f"  Top-{k}: "
                f"{data['category_matches']}/{k} "
                f"category matches | "
                f"Success: "
                f"{'YES' if data['success'] else 'NO'} | "
                f"Avg similarity: "
                f"{data['average_similarity']:.4f}"
            )

        else:

            print(
                f"  Top-{k}: "
                f"Category evaluation unavailable | "
                f"Avg similarity: "
                f"{data['average_similarity']:.4f}"
            )


# ============================================================
# INTERPRETATION
# ============================================================

print("\n")
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

if researchers_with_categories > 0:

    top5 = overall_results.get(5)

    if top5:

        success_rate = (
            top5["researchers_with_match"]
            / researchers_with_categories
        )

        individual_rate = (
            top5["category_matches"]
            / top5["recommendations"]
            if top5["recommendations"] > 0
            else 0
        )

        print(
            f"\nTop-5 success rate: "
            f"{success_rate * 100:.2f}%"
        )

        print(
            f"Top-5 individual category match rate: "
            f"{individual_rate * 100:.2f}%"
        )

        if success_rate >= 0.80:

            print(
                "\nStrong researcher-to-paper "
                "category retrieval performance."
            )

        elif success_rate >= 0.60:

            print(
                "\nModerate researcher-to-paper "
                "category retrieval performance."
            )

        else:

            print(
                "\nWeak researcher-to-paper "
                "category retrieval performance."
            )

else:

    print(
        "\nSimilarity-based evaluation completed."
    )

    print(
        "Category-based evaluation could not be "
        "performed because researcher profiles "
        "do not contain categories."
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print(
    "RESEARCHER -> PAPER EVALUATION COMPLETE"
)
print("=" * 60)