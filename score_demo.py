#!/usr/bin/env python3
# pip install transformers torch pandas

import argparse
import pandas as pd
from transformers import pipeline

COL_R1 = "Review 1"
COL_R2 = "Review 2"
COL_F1 = "APP Features 1"
COL_F2 = "App Features 2"

def load_nli(model_name: str, device: int):
    return pipeline("text-classification", model=model_name, return_all_scores=True, device=device)

def entail_prob(nli, premise: str, hypothesis: str) -> float:
    out = nli(f"{premise} </s></s> {hypothesis}")[0]
    d = {x["label"].lower(): float(x["score"]) for x in out}
    # if model returns LABEL_0/1/2 format
    if "label_2" in d:
        return d["label_2"]  # entailment
    return d["entailment"]

def H(feature: str) -> str:
    return f"This sentence is about: {feature}."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--th", type=float, required=True, help="Threshold")
    ap.add_argument("--model", default="roberta-large-mnli")
    ap.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 GPU0, ...")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    nli = load_nli(args.model, args.device)

    scores = []
    preds = []

    for _, row in df.iterrows():
        r1 = "" if pd.isna(row.get(COL_R1)) else str(row.get(COL_R1))
        r2 = "" if pd.isna(row.get(COL_R2)) else str(row.get(COL_R2))
        f1 = "" if pd.isna(row.get(COL_F1)) else str(row.get(COL_F1))
        f2 = "" if pd.isna(row.get(COL_F2)) else str(row.get(COL_F2))

        s12 = entail_prob(nli, r1, H(f2))   # Review1 supports Feature2?
        s21 = entail_prob(nli, r2, H(f1))   # Review2 supports Feature1?
        score = min(s12, s21)

        scores.append(score)
        preds.append(1 if score >= args.th else 0)

    df["score"] = scores
    df["pred"] = preds
    df.to_csv(args.out, index=False)

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
