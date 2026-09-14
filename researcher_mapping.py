import json
import csv
from pathlib import Path

INPUT_FILE = Path("data/raw/arxiv_raw.json")
OUTPUT_FILE = Path("data/processed/researcher_papers.csv")

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    papers = json.load(file)

researcher_papers = []

for paper in papers:
    paper_id = paper["paper_id"]

    for author in paper["authors"]:
        researcher_papers.append({
            "researcher_id": author,
            "researcher_name": author,
            "paper_id": paper_id
        })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
    fieldnames = [
        "researcher_id",
        "researcher_name",
        "paper_id"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(researcher_papers)

print(
    f"Saved {len(researcher_papers)} researcher-paper relationships "
    f"to {OUTPUT_FILE}"
)