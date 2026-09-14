import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("PAPER RECOMMENDATION BASELINE EVALUATION")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load papers
# ---------------------------------------------------------

path = "data/processed/papers_clean.csv"

df = pd.read_csv(path)

print(f"\nNumber of papers: {len(df)}")

# ---------------------------------------------------------
# 2. Prepare text
# ---------------------------------------------------------

texts = (
    df["title"].fillna("") + ". " +
    df["abstract"].fillna("")
).tolist()

# ---------------------------------------------------------
# 3. Load pretrained embedding model
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 4. Generate embeddings
# ---------------------------------------------------------

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embeddings generated successfully!")
print(f"Embedding shape: {embeddings.shape}")

# ---------------------------------------------------------
# 5. Calculate similarity matrix
# ---------------------------------------------------------

similarities = cosine_similarity(embeddings)

# ---------------------------------------------------------
# 6. Evaluate selected papers
# ---------------------------------------------------------

test_indices = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]

print("\n" + "=" * 60)
print("RECOMMENDATION RESULTS")
print("=" * 60)

for index in test_indices:

    print("\n" + "-" * 60)
    print(f"TARGET PAPER #{index + 1}")
    print("-" * 60)

    print(f"Title: {df.iloc[index]['title']}")
    print(f"Categories: {df.iloc[index]['categories']}")

    # Get similarity scores
    scores = similarities[index].copy()

    # Don't recommend the paper itself
    scores[index] = -1

    # Get top 5
    top_indices = np.argsort(scores)[::-1][:5]

    print("\nTop 5 recommendations:")

    for rank, similar_index in enumerate(top_indices, start=1):

        print(f"\n{rank}. {df.iloc[similar_index]['title']}")
        print(f"   Similarity: {scores[similar_index]:.4f}")
        print(f"   Categories: {df.iloc[similar_index]['categories']}")

print("\n" + "=" * 60)
print("BASELINE EVALUATION COMPLETE")
print("=" * 60)