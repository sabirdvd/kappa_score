# pip install transformers torch pandas

import argparse
import pandas as pd
from transformers import pipeline

def load_nli(model_name: str, device: int):
    return pipeline("text-classification", model=model_name, return_all_scores=True, device=device)

def entail_prob(nli, premise: str, hypothesis: str) -> float:
    out = nli(f"{premise} </s></s> {hypothesis}")[0]
    d = {x["label"].lower(): float(x["score"]) for x in out}
    if "label_2" in d:
        return d["label_2"]
    return d["entailment"]

def H(feature: str) -> str:
    return f"This sentence is about: {feature}."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--th", type=float, required=True)
    ap.add_argument("--model", default="roberta-large-mnli")
    ap.add_argument("--device", type=int, default=-1)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    nli = load_nli(args.model, args.device)

    scores, preds = [], []

    for _, row in df.iterrows():
        r1 = "" if pd.isna(row.get("Review 1")) else str(row.get("Review 1"))
        r2 = "" if pd.isna(row.get("Review 2")) else str(row.get("Review 2"))
        f1 = "" if pd.isna(row.get("APP Features 1")) else str(row.get("APP Features 1"))
        f2 = "" if pd.isna(row.get("App Features 2")) else str(row.get("App Features 2"))

        s12 = entail_prob(nli, r1, H(f2))
        s21 = entail_prob(nli, r2, H(f1))
        score = min(s12, s21)

        scores.append(score)
        preds.append(1 if score >= args.th else 0)

    df["score"] = scores
    df["pred"] = preds
    df.to_csv(args.out, index=False)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
