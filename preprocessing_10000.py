import json
import csv
import re
from pathlib import Path
INPUT_FILE = Path("data/raw/arxiv_raw_10000.json")
OUTPUT_FILE = Path("data/processed/papers_clean_10000.csv")
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    papers = json.load(file)

def clean_text(text):
    # Replace newlines and tabs with a space
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from the beginning and end
    text = text.strip()

    return text

cleaned_papers = []

for paper in papers:
    cleaned_paper = {
        "paper_id": paper["paper_id"],
        "title": clean_text(paper["title"]),
        "abstract": clean_text(paper["abstract"]),
        "published": paper["published"],
        "categories": ";".join(paper["categories"])
    }

    cleaned_papers.append(cleaned_paper)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
    fieldnames = [
        "paper_id",
        "title",
        "abstract",
        "published",
        "categories"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(cleaned_papers)

print(f"Saved {len(cleaned_papers)} cleaned papers to {OUTPUT_FILE}")