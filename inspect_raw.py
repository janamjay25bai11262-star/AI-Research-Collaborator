import json
from collections import Counter
from pathlib import Path

# Location of raw dataset
INPUT_FILE = Path("data/raw/arxiv_raw.json")

# Load raw data
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    papers = json.load(file)

print("=" * 60)
print("RAW DATASET INSPECTION")
print("=" * 60)

# --------------------------------------------------
# 1. Number of papers
# --------------------------------------------------

print("\n1. NUMBER OF PAPERS")
print("Total papers:", len(papers))


# --------------------------------------------------
# 2. Missing values
# --------------------------------------------------

print("\n2. MISSING VALUES")

fields = [
    "paper_id",
    "title",
    "abstract",
    "authors",
    "published",
    "updated",
    "categories"
]

for field in fields:
    missing = sum(
        1 for paper in papers
        if not paper.get(field)
    )

    print(f"{field}: {missing} missing")


# --------------------------------------------------
# 3. Duplicate paper IDs
# --------------------------------------------------

print("\n3. DUPLICATE PAPER IDs")

paper_ids = [
    paper.get("paper_id")
    for paper in papers
]

duplicate_ids = len(paper_ids) - len(set(paper_ids))

print("Duplicate paper IDs:", duplicate_ids)


# --------------------------------------------------
# 4. Duplicate titles
# --------------------------------------------------

print("\n4. DUPLICATE TITLES")

titles = [
    paper.get("title", "").strip().lower()
    for paper in papers
]

title_counts = Counter(titles)

duplicate_titles = {
    title: count
    for title, count in title_counts.items()
    if count > 1
}

print("Number of duplicated titles:", len(duplicate_titles))

if duplicate_titles:
    for title, count in duplicate_titles.items():
        print(f"  {count} times: {title}")


# --------------------------------------------------
# 5. Author statistics
# --------------------------------------------------

print("\n5. AUTHOR STATISTICS")

all_authors = []

for paper in papers:
    all_authors.extend(paper.get("authors", []))

unique_authors = set(all_authors)

print("Total author appearances:", len(all_authors))
print("Unique authors:", len(unique_authors))


# --------------------------------------------------
# 6. Researchers appearing on multiple papers
# --------------------------------------------------

print("\n6. AUTHORS WITH MULTIPLE PAPERS")

author_paper_counts = Counter(all_authors)

repeat_authors = {
    author: count
    for author, count in author_paper_counts.items()
    if count > 1
}

print("Authors appearing on more than one paper:",
      len(repeat_authors))

for author, count in sorted(
    repeat_authors.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:
    print(f"  {author}: {count} papers")


# --------------------------------------------------
# 7. Abstract length
# --------------------------------------------------

print("\n7. ABSTRACT LENGTH")

abstract_lengths = [
    len(paper.get("abstract", ""))
    for paper in papers
]

if abstract_lengths:
    print("Shortest abstract:",
          min(abstract_lengths), "characters")

    print("Longest abstract:",
          max(abstract_lengths), "characters")

    print("Average abstract:",
          round(sum(abstract_lengths) / len(abstract_lengths), 2),
          "characters")


# --------------------------------------------------
# 8. Number of authors per paper
# --------------------------------------------------

print("\n8. AUTHORS PER PAPER")

authors_per_paper = [
    len(paper.get("authors", []))
    for paper in papers
]

if authors_per_paper:
    print("Minimum authors:",
          min(authors_per_paper))

    print("Maximum authors:",
          max(authors_per_paper))

    print("Average authors:",
          round(
              sum(authors_per_paper) / len(authors_per_paper),
              2
          ))


# --------------------------------------------------
# 9. Categories
# --------------------------------------------------

print("\n9. RESEARCH CATEGORIES")

all_categories = []

for paper in papers:
    all_categories.extend(
        paper.get("categories", [])
    )

category_counts = Counter(all_categories)

print("Different categories:",
      len(category_counts))

for category, count in category_counts.most_common():
    print(f"  {category}: {count}")


# --------------------------------------------------
# 10. Show first 3 papers
# --------------------------------------------------

print("\n10. SAMPLE PAPERS")

for i, paper in enumerate(papers[:3], start=1):

    print(f"\n--- Paper {i} ---")

    print("Title:")
    print(paper.get("title"))

    print("Authors:")
    print(paper.get("authors"))

    print("Categories:")
    print(paper.get("categories"))

    print("Abstract preview:")
    print(paper.get("abstract", "")[:300])


print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)