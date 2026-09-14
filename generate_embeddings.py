import csv
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("data/processed/researcher_profiles_clean.csv")
OUTPUT_FILE = Path("data/processed/researcher_embeddings.npy")

print("Loading researcher profiles...")

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    profiles = list(csv.DictReader(file))

print(f"Researchers found: {len(profiles)}")

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

research_texts = [
    profile["research_text"]
    for profile in profiles
]

embeddings = model.encode(
    research_texts,
    show_progress_bar=True
)

embeddings = np.array(embeddings)

np.save(OUTPUT_FILE, embeddings)

print("Embeddings generated successfully!")
print("Number of embeddings:", embeddings.shape[0])
print("Embedding dimensions:", embeddings.shape[1])
print("Saved to:", OUTPUT_FILE)