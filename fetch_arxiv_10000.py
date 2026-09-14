import arxiv
import json
from pathlib import Path

# Search for 10,000 AI research papers
search = arxiv.Search(
    query="cat:cs.AI",
    max_results=10000,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)

client = arxiv.Client(
    page_size=100,
    delay_seconds=3.0,
    num_retries=3
)

papers = []

print("Collecting 10,000 arXiv AI papers...")
print("Please be patient; this may take some time.")

for i, paper in enumerate(client.results(search), start=1):

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

    if i % 100 == 0:
        print(f"Collected {i} papers...")

    if i >= 10000:
        break

# Create raw-data folder
output_folder = Path("data/raw")
output_folder.mkdir(parents=True, exist_ok=True)

# IMPORTANT: separate file from the 100-paper dataset
output_file = output_folder / "arxiv_raw_10000.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(papers, file, indent=2, ensure_ascii=False)

print()
print("=" * 60)
print("COLLECTION COMPLETE")
print("=" * 60)
print(f"Total papers collected: {len(papers)}")
print(f"Saved to: {output_file}")