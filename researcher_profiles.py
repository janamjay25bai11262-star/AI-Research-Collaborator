import csv
from pathlib import Path
from collections import defaultdict

PAPERS_FILE = Path("data/processed/papers_clean.csv")
MAPPING_FILE = Path("data/processed/researcher_papers.csv")
OUTPUT_FILE = Path("data/processed/researcher_profiles.csv")


with open(PAPERS_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

with open(MAPPING_FILE, "r", encoding="utf-8") as file:
    researcher_papers = list(csv.DictReader(file))

paper_lookup = {}

for paper in papers:
    paper_lookup[paper["paper_id"]] = paper

researcher_data = defaultdict(list)

for row in researcher_papers:
    researcher_id = row["researcher_id"]
    paper_id = row["paper_id"]

    researcher_data[researcher_id].append(paper_id)

researcher_profiles = []

for researcher_id, paper_ids in researcher_data.items():

    categories = set()
    research_text_parts = []

    for paper_id in paper_ids:
        paper = paper_lookup.get(paper_id)

        if paper is None:
            continue

        # Collect categories
        for category in paper["categories"].split(";"):
            categories.add(category)

        # Combine title and abstract
        research_text_parts.append(
            paper["title"] + ". " + paper["abstract"]
        )

    researcher_profiles.append({
        "researcher_id": researcher_id,
        "researcher_name": researcher_id,
        "paper_count": len(paper_ids),
        "paper_ids": ";".join(paper_ids),
        "categories": ";".join(sorted(categories)),
        "research_text": " ".join(research_text_parts)
    })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
    fieldnames = [
        "researcher_id",
        "researcher_name",
        "paper_count",
        "paper_ids",
        "categories",
        "research_text"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(researcher_profiles)

print(
    f"Saved {len(researcher_profiles)} researcher profiles "
    f"to {OUTPUT_FILE}"
)