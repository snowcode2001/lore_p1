"""Download models before running the app."""

from transformers import pipeline
from sentence_transformers import SentenceTransformer

print("Downloading zero-shot classifier (bart-large-mnli)...")
pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("Downloading embedding model (all-MiniLM-L6-v2)...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading sentiment analyzer (distilbert-base-uncased-finetuned-sst-2-english)")
pipeline(
    "sentiment-analysis", 
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    return_all_scores=True,
)

print("Done. Models cached in ~/.cache/huggingface/")
