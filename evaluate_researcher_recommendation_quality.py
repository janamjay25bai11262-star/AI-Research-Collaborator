import csv
import numpy as np

RF = "data/processed/researcher_profiles_clean.csv"
PF = "data/processed/papers_clean_10000.csv"
RE = "data/processed/researcher_embeddings.npy"
PE = "data/processed/paper_embeddings_10000.npy"

with open(RF, encoding="utf-8") as f:
    researchers = list(csv.DictReader(f))

with open(PF, encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

r = np.load(RE)
p = np.load(PE)

r /= np.linalg.norm(r, axis=1, keepdims=True)
p /= np.linalg.norm(p, axis=1, keepdims=True)

print("=" * 60)
print("RESEARCHER -> PAPER RECOMMENDATION QUALITY")
print("=" * 60)
print(f"Researchers: {len(researchers)}")
print(f"Papers: {len(papers)}")

# Researcher category fields
def cats(x):
    for key in ("categories", "research_categories", "category"):
        if key in x and x[key]:
            return set(x[key].replace(",", ";").split(";"))
    return set()

ks = [1, 5, 10, 20]
hits = {k: 0 for k in ks}
total = {k: 0 for k in ks}
similarity = {k: [] for k in ks}

for i, researcher in enumerate(researchers):

    rc = cats(researcher)

    if not rc:
        continue

    scores = p @ r[i]
    ranked = np.argsort(scores)[::-1]

    for k in ks:
        top = ranked[:k]

        for j in top:
            pc = cats(papers[j])

            if rc & pc:
                hits[k] += 1

        total[k] += k
        similarity[k].append(np.mean(scores[top]))

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

for k in ks:
    success = sum(
        1 for i, researcher in enumerate(researchers)
        if cats(researcher)
        and any(
            cats(papers[j]) & cats(researcher)
            for j in np.argsort(p @ r[i])[::-1][:k]
        )
    )

    match_rate = hits[k] / total[k] * 100 if total[k] else 0
    success_rate = success / sum(bool(cats(x)) for x in researchers) * 100

    print(f"\nTOP-{k}")
    print("-" * 40)
    print(f"Researchers with category match: {success}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Individual category match rate: {match_rate:.2f}%")
    print(f"Average cosine similarity: {np.mean(similarity[k]):.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("Higher category-match rates indicate stronger")
print("researcher-to-paper recommendation relevance.")
print("\nEVALUATION COMPLETE")