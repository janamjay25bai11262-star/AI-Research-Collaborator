import json
from pathlib import Path

INPUT_FILE = Path("data/raw/arxiv_raw.json")

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    papers = json.load(file)

print("=" * 60)
print("DETAILED RAW DATA INSPECTION")
print("=" * 60)

# --------------------------------------------------
# 1. AUTHOR NAME INSPECTION
# --------------------------------------------------

print("\n1. AUTHOR NAMES")

all_authors = []

for paper in papers:
    all_authors.extend(paper["authors"])

print("Total author names:", len(all_authors))

print("\nFirst 20 author names:")

for author in all_authors[:20]:
    print("-", author)


# --------------------------------------------------
# 2. TITLE INSPECTION
# --------------------------------------------------

print("\n2. TITLE INSPECTION")

print("\nFirst 10 titles:")

for i, paper in enumerate(papers[:10], start=1):
    print(f"{i}. {paper['title']}")


# --------------------------------------------------
# 3. ABSTRACT INSPECTION
# --------------------------------------------------

print("\n3. ABSTRACT INSPECTION")

print("\nFirst 3 abstracts:\n")

for i, paper in enumerate(papers[:3], start=1):

    print(f"--- Abstract {i} ---")

    abstract = paper["abstract"]

    print(abstract)

    print()


# --------------------------------------------------
# 4. CHECK FOR NEWLINES
# --------------------------------------------------

print("\n4. NEWLINE CHECK")

abstracts_with_newlines = sum(
    "\n" in paper["abstract"]
    for paper in papers
)

titles_with_newlines = sum(
    "\n" in paper["title"]
    for paper in papers
)

print("Abstracts containing newlines:",
      abstracts_with_newlines)

print("Titles containing newlines:",
      titles_with_newlines)


# --------------------------------------------------
# 5. CHECK FOR EXTRA SPACES
# --------------------------------------------------

print("\n5. EXTRA SPACE CHECK")

abstracts_with_double_spaces = sum(
    "  " in paper["abstract"]
    for paper in papers
)

titles_with_double_spaces = sum(
    "  " in paper["title"]
    for paper in papers
)

print("Abstracts containing double spaces:",
      abstracts_with_double_spaces)

print("Titles containing double spaces:",
      titles_with_double_spaces)


# --------------------------------------------------
# 6. CHECK EMPTY AUTHOR NAMES
# --------------------------------------------------

print("\n6. EMPTY AUTHOR NAME CHECK")

empty_authors = sum(
    not author.strip()
    for author in all_authors
)

print("Empty author names:", empty_authors)


print("\n" + "=" * 60)
print("DETAILED INSPECTION COMPLETE")
print("=" * 60)