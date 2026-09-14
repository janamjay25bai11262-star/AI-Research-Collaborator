import csv
import re
from pathlib import Path
INPUT_FILE = Path("data/processed/researcher_profiles.csv")
OUTPUT_FILE = Path("data/processed/researcher_profiles_clean.csv")
def clean_text(text):
    # Remove simple LaTeX commands such as \textbf{...}
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)

    # Remove remaining LaTeX commands such as \alpha, \cite, etc.
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    # Remove LaTeX math delimiters
    text = text.replace("$", " ")

    # Convert everything to lowercase
    text = text.lower()

    # Replace repeated whitespace with one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    profiles = list(csv.DictReader(file))

cleaned_profiles = []

for profile in profiles:
    cleaned_profile = profile.copy()

    cleaned_profile["research_text"] = clean_text(
        profile["research_text"]
    )

    cleaned_profiles.append(cleaned_profile)

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
    writer.writerows(cleaned_profiles)

print(
    f"Saved {len(cleaned_profiles)} cleaned researcher profiles "
    f"to {OUTPUT_FILE}"
)