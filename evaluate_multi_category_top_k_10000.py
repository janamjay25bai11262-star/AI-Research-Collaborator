import csv
import numpy as np

PAPER_FILE = "data/processed/papers_clean_10000.csv"
EMBEDDING_FILE = "data/processed/paper_embeddings_10000.npy"

with open(PAPER_FILE, encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

embeddings = np.load(EMBEDDING_FILE)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

print("=" * 60)
print("MULTI-CATEGORY TOP-K EVALUATION - 10,000 PAPERS")
print("=" * 60)

print(f"Papers: {len(papers)}")
print(f"Embeddings: {embeddings.shape}")

# Enter multiple categories
raw = input("\nEnter categories (example: cs.LG,cs.AI,cs.CL): ")
target_categories = {
    x.strip() for x in raw.replace(";", ",").split(",") if x.strip()
}

print("\nTarget categories:", ", ".join(sorted(target_categories)))

# Target papers: papers containing at least one requested category
targets = []

for i, paper in enumerate(papers):
    categories = {
        x.strip()
        for x in paper.get("categories", "").replace(",", ";").split(";")
        if x.strip()
    }

    if categories & target_categories:
        targets.append(i)

print(f"Papers matching at least one target category: {len(targets)}")

# Evaluate representative target papers
sample = targets[::max(1, len(targets) // 6)][:6]

ks = [1, 3, 5, 10, 20]

for target in sample:

    scores = embeddings @ embeddings[target]
    scores[target] = -1

    ranked = np.argsort(scores)[::-1]

    print("\n" + "=" * 60)
    print("TARGET PAPER")
    print("=" * 60)
    print(papers[target]["title"])
    print("Categories:", papers[target].get("categories", ""))

    for k in ks:

        top = ranked[:k]
        total_category_matches = 0
        perfect_matches = 0

        for idx in top:
            paper_categories = {
                x.strip()
                for x in papers[idx].get("categories", "")
                .replace(",", ";").split(";")
                if x.strip()
            }

            matches = target_categories & paper_categories
            total_category_matches += len(matches)

            if target_categories.issubset(paper_categories):
                perfect_matches += 1

        possible = k * len(target_categories)
        percentage = (
            total_category_matches / possible * 100
            if possible else 0
        )

        print(
            f"Top-{k}: "
            f"{total_category_matches}/{possible} "
            f"category matches | "
            f"Category coverage: {percentage:.2f}% | "
            f"All-category papers: {perfect_matches}"
        )

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)