import json
from pathlib import Path
from collections import Counter

# ============================================================
# INSPECT 10,000-PAPER ARXIV DATASET
# ============================================================

input_file = Path("data/raw/arxiv_raw_10000.json")

print("=" * 60)
print("10,000-PAPER DATASET INSPECTION")
print("=" * 60)

# Check file exists
if not input_file.exists():
    print(f"ERROR: File not found: {input_file}")
    raise SystemExit(1)

# Load dataset
with open(input_file, "r", encoding="utf-8") as file:
    papers = json.load(file)

print(f"\nTotal papers: {len(papers)}")

# ------------------------------------------------------------
# BASIC FIELD CHECKS
# ------------------------------------------------------------

required_fields = [
    "paper_id",
    "title",
    "abstract",
    "authors",
    "published",
    "updated",
    "categories"
]

print("\n" + "=" * 60)
print("MISSING FIELD CHECK")
print("=" * 60)

for field in required_fields:
    missing = sum(
        1 for paper in papers
        if field not in paper or paper[field] is None or paper[field] == ""
    )

    print(f"{field}: {missing} missing")

# ------------------------------------------------------------
# PAPER ID DUPLICATES
# ------------------------------------------------------------

paper_ids = [paper.get("paper_id") for paper in papers]
id_counts = Counter(paper_ids)

duplicate_ids = {
    paper_id: count
    for paper_id, count in id_counts.items()
    if count > 1
}

print("\n" + "=" * 60)
print("DUPLICATE PAPER ID CHECK")
print("=" * 60)

print(f"Unique paper IDs: {len(id_counts)}")
print(f"Duplicate paper IDs: {len(duplicate_ids)}")

if duplicate_ids:
    print("\nDuplicate IDs:")
    for paper_id, count in list(duplicate_ids.items())[:10]:
        print(f"{paper_id}: {count}")

# ------------------------------------------------------------
# DUPLICATE TITLES
# ------------------------------------------------------------

titles = [
    paper.get("title", "").strip().lower()
    for paper in papers
]

title_counts = Counter(titles)

duplicate_titles = {
    title: count
    for title, count in title_counts.items()
    if title and count > 1
}

print("\n" + "=" * 60)
print("DUPLICATE TITLE CHECK")
print("=" * 60)

print(f"Unique titles: {len(title_counts)}")
print(f"Duplicate titles: {len(duplicate_titles)}")

if duplicate_titles:
    print("\nExamples:")
    for title, count in list(duplicate_titles.items())[:10]:
        print(f"{count}x - {title[:100]}")

# ------------------------------------------------------------
# AUTHOR STATISTICS
# ------------------------------------------------------------

author_counts = [
    len(paper.get("authors", []))
    for paper in papers
]

print("\n" + "=" * 60)
print("AUTHOR STATISTICS")
print("=" * 60)

print(f"Average authors per paper: {sum(author_counts) / len(author_counts):.2f}")
print(f"Minimum authors: {min(author_counts)}")
print(f"Maximum authors: {max(author_counts)}")
print(f"Papers with no authors: {sum(1 for x in author_counts if x == 0)}")

# ------------------------------------------------------------
# ABSTRACT STATISTICS
# ------------------------------------------------------------

abstract_lengths = [
    len(paper.get("abstract", ""))
    for paper in papers
]

print("\n" + "=" * 60)
print("ABSTRACT STATISTICS")
print("=" * 60)

print(f"Average abstract length: {sum(abstract_lengths) / len(abstract_lengths):.0f} characters")
print(f"Shortest abstract: {min(abstract_lengths)} characters")
print(f"Longest abstract: {max(abstract_lengths)} characters")
print(
    f"Papers with empty abstracts: "
    f"{sum(1 for x in abstract_lengths if x == 0)}"
)

# ------------------------------------------------------------
# CATEGORY STATISTICS
# ------------------------------------------------------------

category_counter = Counter()

for paper in papers:
    for category in paper.get("categories", []):
        category_counter[category] += 1

print("\n" + "=" * 60)
print("CATEGORY DISTRIBUTION")
print("=" * 60)

print(f"Unique categories: {len(category_counter)}")

print("\nTop 20 categories:")

for category, count in category_counter.most_common(20):
    print(f"{category}: {count}")

# ------------------------------------------------------------
# SAMPLE PAPERS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE PAPERS")
print("=" * 60)

for i, paper in enumerate(papers[:5], start=1):

    print(f"\n--- PAPER {i} ---")
    print(f"ID: {paper.get('paper_id')}")
    print(f"Title: {paper.get('title')}")
    print(f"Authors: {len(paper.get('authors', []))}")
    print(f"Published: {paper.get('published')}")
    print(f"Categories: {paper.get('categories')}")
    print(f"Abstract preview: {paper.get('abstract', '')[:300]}...")

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)

print(f"Total papers checked: {len(papers)}")
print(f"Unique paper IDs: {len(id_counts)}")
print(f"Duplicate paper IDs: {len(duplicate_ids)}")
print(f"Duplicate titles: {len(duplicate_titles)}")

print("\nThe 10,000-paper dataset is ready for review.")