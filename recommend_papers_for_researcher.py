import csv
import numpy as np

RESEARCHER_FILE = "data/processed/researcher_profiles_clean.csv"
PAPER_FILE = "data/processed/papers_clean_10000.csv"
RESEARCHER_EMB = "data/processed/researcher_embeddings.npy"
PAPER_EMB = "data/processed/paper_embeddings_10000.npy"

# Load data
with open(RESEARCHER_FILE, encoding="utf-8") as f:
    researchers = list(csv.DictReader(f))

with open(PAPER_FILE, encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

r_emb = np.load(RESEARCHER_EMB)
p_emb = np.load(PAPER_EMB)

# Normalize embeddings
r_emb /= np.linalg.norm(r_emb, axis=1, keepdims=True)
p_emb /= np.linalg.norm(p_emb, axis=1, keepdims=True)

print("=" * 60)
print("RESEARCHER -> PAPER RECOMMENDER")
print("=" * 60)
print(f"Researchers: {len(researchers)}")
print(f"Papers: {len(papers)}")

name = input("\nEnter researcher name: ").strip()

# Find researcher
matches = [
    i for i, r in enumerate(researchers)
    if r["researcher_name"].strip().lower() == name.lower()
]

if not matches:
    print("\nResearcher not found.")
    raise SystemExit

i = matches[0]
researcher = researchers[i]

# Similarity against all 10,000 papers
scores = p_emb @ r_emb[i]

# Rank highest first
ranked = np.argsort(scores)[::-1]

print("\n" + "=" * 60)
print("RESEARCHER")
print("=" * 60)
print(f"Name: {researcher['researcher_name']}")

print("\n" + "=" * 60)
print("TOP 10 RECOMMENDED PAPERS")
print("=" * 60)

for rank, p in enumerate(ranked[:10], 1):
    print(f"\n{rank}. {papers[p]['title']}")
    print(f"   Similarity: {scores[p]:.4f}")
    print(f"   Categories: {papers[p]['categories']}")
    print(f"   Published: {papers[p]['published']}")
    print(f"   Paper ID: {papers[p]['paper_id']}")

print("\n" + "=" * 60)
print("RECOMMENDATION COMPLETE")
print("=" * 60)