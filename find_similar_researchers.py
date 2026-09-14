import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROFILE_FILE = "data/processed/researcher_profiles_clean.csv"
EMBEDDING_FILE = "data/processed/researcher_embeddings.npy"

# Load researcher profiles
with open(PROFILE_FILE, "r", encoding="utf-8") as file:
    researchers = list(csv.DictReader(file))

# Load embeddings
embeddings = np.load(EMBEDDING_FILE)

# Calculate similarity between every researcher
similarity_matrix = cosine_similarity(embeddings)

# Choose one researcher to test
researcher_index = 0

researcher_name = researchers[researcher_index]["researcher_name"]

print("=" * 60)
print("RESEARCHER SIMILARITY TEST")
print("=" * 60)

print(f"\nTarget researcher: {researcher_name}")

# Get similarity scores for this researcher
scores = similarity_matrix[researcher_index]

# Sort from highest to lowest
similar_indices = np.argsort(scores)[::-1]

print("\nTop 5 similar researchers:\n")

count = 0

for index in similar_indices:

    # Don't recommend the researcher to themselves
    if index == researcher_index:
        continue

    print(
        f"{count + 1}. "
        f"{researchers[index]['researcher_name']} "
        f"(similarity: {scores[index]:.4f})"
    )

    count += 1

    if count == 5:
        break