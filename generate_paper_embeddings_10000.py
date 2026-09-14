import csv
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("data/processed/papers_clean_10000.csv")
OUTPUT_FILE = Path("data/processed/paper_embeddings_10000.npy")

print("=" * 60)
print("GENERATING 10,000 PAPER EMBEDDINGS")
print("=" * 60)

print("\nLoading papers...")

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    papers = list(csv.DictReader(file))

print(f"Papers found: {len(papers)}")

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nGenerating embeddings...")

# Combine title and abstract so the embedding represents
# both the paper topic and its description.
paper_texts = [
    f"{paper['title']}. {paper['abstract']}"
    for paper in papers
]

embeddings = model.encode(
    paper_texts,
    batch_size=32,
    show_progress_bar=True
)

embeddings = np.array(embeddings)

print("\nEmbeddings generated successfully!")
print("Number of embeddings:", embeddings.shape[0])
print("Embedding dimensions:", embeddings.shape[1])

np.save(OUTPUT_FILE, embeddings)

print("Saved to:", OUTPUT_FILE)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)