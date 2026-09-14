import csv
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

PAPERS_FILE = Path("data/processed/papers_clean_10000.csv")
EMBEDDINGS_FILE = Path("data/processed/paper_embeddings_10000.npy")

print("=" * 60)
print("10,000-PAPER SIMILARITY TEST")
print("=" * 60)

print("\nLoading papers...")

with open(PAPERS_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

print(f"Papers loaded: {len(papers)}")

print("\nLoading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embeddings loaded: {embeddings.shape}")

# Check that papers and embeddings correspond
if len(papers) != embeddings.shape[0]:
    raise ValueError(
        f"Mismatch: {len(papers)} papers but "
        f"{embeddings.shape[0]} embeddings"
    )

# Test several papers spread throughout the dataset
test_indices = [0, 999, 1999, 4999, 9999]

print("\n" + "=" * 60)
print("SIMILARITY RESULTS")
print("=" * 60)

for target_index in test_indices:

    target_paper = papers[target_index]
    target_embedding = embeddings[target_index].reshape(1, -1)

    similarities = cosine_similarity(
        target_embedding,
        embeddings
    )[0]

    # Don't recommend the target paper itself
    similarities[target_index] = -1

    top_indices = np.argsort(similarities)[::-1][:5]

    print("\n" + "-" * 60)
    print(f"TARGET PAPER #{target_index + 1}")
    print("-" * 60)

    print("Title:")
    print(target_paper["title"])

    print("\nCategories:")
    print(target_paper["categories"])

    print("\nTop 5 similar papers:")

    for rank, index in enumerate(top_indices, start=1):
        paper = papers[index]

        print(f"\n{rank}. {paper['title']}")
        print(f"   Similarity: {similarities[index]:.4f}")
        print(f"   Categories: {paper['categories']}")

print("\n" + "=" * 60)
print("SIMILARITY TEST COMPLETE")
print("=" * 60)