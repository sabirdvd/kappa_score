# !pip install pandas numpy torch transformers scikit-learn sentence-transformers

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sentence_transformers import CrossEncoder

INPUT_CSV = "chatgpt_vs_gemini_d1.csv"
SCORED_OUTPUT_CSV = "scored_multi_model.csv"
THRESHOLD_REPORT_CSV = "threshold_report.csv"

MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REGISTRY = {
    "roberta_mnli": {
        "type": "hf_nli",
        "name": "roberta-large-mnli",
    },
    "deberta_mnli": {
        "type": "hf_nli",
        "name": "microsoft/deberta-large-mnli",
    },
    "modernbert_nli_tasksource": {
        "type": "hf_nli",
        "name": "tasksource/ModernBERT-base-nli",
    },
    "modernbert_nli_mrm8488": {
        "type": "hf_nli",
        "name": "mrm8488/ModernBERT-base-ft-all-nli",
    },
    "crossencoder_stsb": {
        "type": "cross_encoder",
        "name": "cross-encoder/stsb-roberta-large",
        "normalize": "none"
    },
}

TH_START, TH_END, TH_STEP = 0.10, 0.90, 0.01
DEFAULT_THRESHOLD = 0.60

def make_hypothesis(feature: str) -> str:
    if pd.isna(feature):
        feature = ""
    return f"This text discusses {str(feature).strip()}."


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tune_threshold(y_true, y_score, start=0.10, end=0.90, step=0.01):
    rows = []
    for th in np.arange(start, end + 1e-12, step):
        y_pred = (y_score >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)
        rows.append({
            "threshold": round(float(th), 3),
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        })
    return pd.DataFrame(rows).sort_values(
        ["f1", "precision", "recall", "accuracy"], ascending=False
    ).reset_index(drop=True)

class HFNLIModel:
    """
    HuggingFace sequence classifier expected to output 3-way NLI logits:
    [contradiction, neutral, entailment] OR equivalent label mapping.
    """
    def __init__(self, model_name: str, max_length=256, device="cpu"):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.entail_idx = self._detect_entailment_index()

    def _detect_entailment_index(self):
        id2label = getattr(self.model.config, "id2label", None)
        if not id2label:
            return 2
        for idx, lbl in id2label.items():
            if "entail" in str(lbl).lower():
                return int(idx)
        
        return 2

    def score(self, premise: str, hypothesis: str) -> float:
        premise = "" if pd.isna(premise) else str(premise)
        hypothesis = "" if pd.isna(hypothesis) else str(hypothesis)

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        return float(probs[self.entail_idx])

class CrossEncoderModel:
    """
    sentence-transformers CrossEncoder wrapper.
    Output normalization:
      - 'sigmoid': applies sigmoid(raw_score)
      - 'none': assumes score already in [0,1]
    """
    def __init__(self, model_name: str, normalize="sigmoid", device="cpu"):
        self.model_name = model_name
        self.normalize = normalize
        self.model = CrossEncoder(model_name, device=device)

    def score(self, premise: str, hypothesis: str) -> float:
        premise = "" if pd.isna(premise) else str(premise)
        hypothesis = "" if pd.isna(hypothesis) else str(hypothesis)

        raw = self.model.predict([(premise, hypothesis)])
        raw_val = float(raw[0])

        if self.normalize == "sigmoid":
            return float(sigmoid(raw_val))
        return float(raw_val)


def build_model(entry: dict, device="cpu"):
    if entry["type"] == "hf_nli":
        return HFNLIModel(entry["name"], max_length=MAX_LENGTH, device=device)
    elif entry["type"] == "cross_encoder":
        return CrossEncoderModel(
            entry["name"],
            normalize=entry.get("normalize", "sigmoid"),
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {entry['type']}")

def bidirectional_score(model_obj, review1, feature1, review2, feature2):
    h1 = make_hypothesis(feature1)
    h2 = make_hypothesis(feature2)

    e12 = model_obj.score(review1, h2)
    e21 = model_obj.score(review2, h1)
    smin = min(e12, e21)

    return h1, h2, e12, e21, smin

def main():
    df = pd.read_csv(INPUT_CSV)

    required = ["Review 1", "APP Features 1", "Review 2", "App Features 2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    report_rows = []
    h_cols_written = False

    for model_key, meta in MODEL_REGISTRY.items():
        print(f"\nLoading {model_key}: {meta['name']} ({meta['type']})")
        model_obj = build_model(meta, device=DEVICE)

        e12_list, e21_list, smin_list = [], [], []
        h1_list, h2_list = [], []

        for _, row in df.iterrows():
            h1, h2, e12, e21, smin = bidirectional_score(
                model_obj,
                row["Review 1"], row["APP Features 1"],
                row["Review 2"], row["App Features 2"]
            )
            h1_list.append(h1)
            h2_list.append(h2)
            e12_list.append(e12)
            e21_list.append(e21)
            smin_list.append(smin)

        if not h_cols_written:
            df["H1"] = h1_list
            df["H2"] = h2_list
            h_cols_written = True

        df[f"e12_{model_key}"] = np.round(e12_list, 6)
        df[f"e21_{model_key}"] = np.round(e21_list, 6)
        df[f"score_min_{model_key}"] = np.round(smin_list, 6)

        if "label" in df.columns:
            val = df.dropna(subset=["label", f"score_min_{model_key}"]).copy()
            val["label"] = val["label"].astype(int)

            tr = tune_threshold(
                y_true=val["label"].values,
                y_score=val[f"score_min_{model_key}"].values,
                start=TH_START, end=TH_END, step=TH_STEP
            )
            best_th = float(tr.iloc[0]["threshold"])
            best = tr.iloc[0].to_dict()

            df[f"pred_{model_key}"] = (df[f"score_min_{model_key}"] >= best_th).astype(int)

            report_rows.append({
                "model_key": model_key,
                "model_name": meta["name"],
                "best_threshold": best_th,
                "best_f1": best["f1"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "best_accuracy": best["accuracy"],
            })
        else:
            df[f"pred_{model_key}"] = (df[f"score_min_{model_key}"] >= DEFAULT_THRESHOLD).astype(int)

    df.to_csv(SCORED_OUTPUT_CSV, index=False)
    print(f"\nSaved: {SCORED_OUTPUT_CSV}")

    if report_rows:
        rep = pd.DataFrame(report_rows).sort_values("best_f1", ascending=False)
        rep.to_csv(THRESHOLD_REPORT_CSV, index=False)
        print(f"Saved: {THRESHOLD_REPORT_CSV}")
        print("\nModel ranking by best_f1:")
        print(rep[["model_key", "best_f1", "best_precision", "best_recall", "best_threshold"]])

if __name__ == "__main__":
    main()