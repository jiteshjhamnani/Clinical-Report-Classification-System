import re
import pickle
from pathlib import Path

import joblib
import textstat
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = "peakyblends/clinical-report-biobert"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
MEDICAL_TERMS_PATH = BASE_DIR / "medical_terms.pkl"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Loading label encoder...")
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("Loading medical vocabulary...")
with open(MEDICAL_TERMS_PATH, "rb") as f:
    raw_terms = pickle.load(f)

# Remove very generic words so the detector is a bit cleaner.
GENERIC_TERMS = {
    "study", "studies", "analysis", "case", "cases", "review", "reports",
    "report", "history", "management", "practice", "materials", "details",
    "information", "discussion", "document", "notes", "note", "follow up",
    "follow-up", "followup", "results", "result", "test", "tests", "testing",
    "procedure", "procedures"
}

medical_terms = {
    str(t).strip().lower()
    for t in raw_terms
    if isinstance(t, str) and str(t).strip() and str(t).strip().lower() not in GENERIC_TERMS
}

model.eval()
print("Everything loaded successfully!")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def medical_score(text: str) -> int:
    """
    Counts unique medical keyword/phrase matches in the input text.
    """
    text_n = _normalize(text)
    matches = set()

    for term in medical_terms:
        term_n = _normalize(term)

        if len(term_n) < 4:
            continue

        # Phrase-like terms: direct substring match
        if (" " in term_n) or ("/" in term_n) or ("-" in term_n):
            if term_n in text_n:
                matches.add(term_n)
        else:
            if re.search(rf"\b{re.escape(term_n)}\b", text_n):
                matches.add(term_n)

    return len(matches)


def predict_report(text: str):
    text = text or ""
    cleaned = text.strip()

    # Basic validation
    if len(cleaned.split()) < 20:
        return {
            "status": "rejected",
            "message": "Input is too short to be a valid medical report."
        }

    # Medical/non-medical check
    score = medical_score(cleaned)
    if score < 3:
        return {
            "status": "rejected",
            "message": "Input does not appear to be a medical report."
        }

    # Readability
    grade = textstat.flesch_kincaid_grade(cleaned)

    if grade < 8:
        complexity = "Easy"
    elif grade < 12:
        complexity = "Moderate"
    else:
        complexity = "Complex"

    # Tokenize
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    top_probs, top_indices = torch.topk(probs, k=3)

    predictions = []
    for idx, prob in zip(top_indices[0], top_probs[0]):
        label = label_encoder.inverse_transform([idx.item()])[0]
        predictions.append({
            "specialty": label.strip(),
            "confidence": round(prob.item() * 100, 2)
        })

    top_prediction = predictions[0]

    return {
        "status": "success",
        "specialty": top_prediction["specialty"],
        "confidence": top_prediction["confidence"],
        "top_predictions": predictions,
        "readability_score": round(grade, 2),
        "complexity": complexity,
        "medical_score": score
    }