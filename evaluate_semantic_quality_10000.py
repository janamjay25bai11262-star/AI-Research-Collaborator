import csv
import numpy as np

PAPER_FILE = "data/processed/papers_clean_10000.csv"
EMBEDDING_FILE = "data/processed/paper_embeddings_10000.npy"

TARGET_INDICES = [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
K_VALUES = [1, 3, 5, 10, 20]

print("=" * 60)
print("10,000-PAPER SEMANTIC RECOMMENDATION QUALITY")
print("=" * 60)

# Load papers
with open(PAPER_FILE, "r", encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

# Load existing embeddings
embeddings = np.load(EMBEDDING_FILE)

print(f"Papers: {len(papers)}")
print(f"Embeddings: {embeddings.shape}")

if len(papers) != len(embeddings):
    raise ValueError("Dataset and embeddings are not aligned!")

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(
    embeddings, axis=1, keepdims=True
)

print("Embeddings normalized.")

all_scores = {k: [] for k in K_VALUES}

for paper_number in TARGET_INDICES:

    i = paper_number - 1
    target = papers[i]

    print("\n" + "=" * 60)
    print(f"TARGET PAPER #{paper_number}")
    print("=" * 60)
    print(target["title"])

    # Semantic similarity against all papers
    scores = embeddings @ embeddings[i]

    # Remove itself
    scores[i] = -1

    ranked = np.argsort(scores)[::-1]

    print("\nTOP RECOMMENDATIONS")
    print("-" * 60)

    for rank, j in enumerate(ranked[:5], 1):

        print(
            f"{rank}. {papers[j]['title']}\n"
            f"   Cosine similarity: {scores[j]:.4f}\n"
            f"   Categories: {papers[j]['categories']}"
        )

    # Top-K average similarity
    for k in K_VALUES:
        top_k_scores = scores[ranked[:k]]
        avg_score = np.mean(top_k_scores)
        all_scores[k].append(avg_score)

        print(
            f"Top-{k} average semantic similarity: "
            f"{avg_score:.4f}"
        )

print("\n" + "=" * 60)
print("OVERALL SEMANTIC QUALITY")
print("=" * 60)

for k in K_VALUES:

    avg = np.mean(all_scores[k])

    print(
        f"Top-{k} average cosine similarity: "
        f"{avg:.4f}"
    )

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

best_k = max(
    K_VALUES,
    key=lambda k: np.mean(all_scores[k])
)

print(
    f"Highest average semantic similarity: "
    f"Top-{best_k}"
)

print(
    f"Average similarity: "
    f"{np.mean(all_scores[best_k]):.4f}"
)

print("\nThis evaluates semantic similarity using the existing")
print("384-dimensional paper embeddings.")
print("\nIMPORTANT: cosine similarity is a semantic ranking signal,")
print("not a human-judged relevance score.")

print("\n" + "=" * 60)
print("SEMANTIC QUALITY EVALUATION COMPLETE")
print("=" * 60)