import arxiv
import json
from pathlib import Path

# Search for 5 AI research papers
search = arxiv.Search(
    query="cat:cs.AI",
    max_results=100
)

client = arxiv.Client()

papers = []

for paper in client.results(search):
    paper_data = {
        "paper_id": paper.entry_id,
        "title": paper.title,
        "abstract": paper.summary,
        "authors": [author.name for author in paper.authors],
        "published": paper.published.isoformat(),
        "updated": paper.updated.isoformat(),
        "categories": paper.categories
    }

    papers.append(paper_data)

# Create the raw-data folder if it doesn't exist
output_folder = Path("data/raw")
output_folder.mkdir(parents=True, exist_ok=True)

# Save the raw data
output_file = output_folder / "arxiv_raw.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(papers, file, indent=2, ensure_ascii=False)

print(f"Saved {len(papers)} papers to {output_file}")