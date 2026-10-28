"""Train and evaluate a TF-IDF + Logistic Regression intent classifier."""
from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    from .preprocessing import clean_text
except ImportError:  # Support running this file directly from the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_support.csv"
MODEL_DIR = ROOT / "models"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"utterance", "intent"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset must contain columns: {sorted(required)}. Missing: {sorted(missing_columns)}")

    before = len(df)
    df = df.dropna(subset=["utterance", "intent"]).copy()
    df["utterance"] = df["utterance"].map(clean_text)
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["utterance"] != "") & (df["intent"] != "")]
    df = df.drop_duplicates(subset=["utterance", "intent"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["utterance"], df["intent"], test_size=0.2, random_state=42, stratify=df["intent"]
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1500, C=4.0, random_state=42)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "chatbot_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    metrics = {
        "source_rows": int(before),
        "usable_rows": int(len(df)),
        "dropped_rows": int(before - len(df)),
        "intent_classes": int(df["intent"].nunique()),
        "test_accuracy": float(accuracy),
        "classification_report": report,
    }
    (MODEL_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Rows in CSV: {before}")
    print(f"Usable rows after cleaning: {len(df)}")
    print(f"Intent classes: {df['intent'].nunique()}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy:.1%})")
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"\nSaved model files to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
