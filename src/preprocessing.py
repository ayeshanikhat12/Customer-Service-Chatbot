"""Lightweight text cleaning helpers for the customer service chatbot."""
import re


def clean_text(text: object) -> str:
    """Normalize a customer utterance for intent classification."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
