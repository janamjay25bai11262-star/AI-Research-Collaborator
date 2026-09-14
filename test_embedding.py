from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Deep learning methods for computer vision."

embedding = model.encode(text)

print("Model loaded successfully!")
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])