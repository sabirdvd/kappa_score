#!/usr/bin/env python3
# pip install transformers torch pandas

import argparse
import pandas as pd
import torch
from transformers import pipeline

def entail_prob(nli, premise: str, hypothesis: str) -> float:
    out = nli(f"{premise} </s></s> {hypothesis}")[0]
    d = {x["label"].lower(): float(x["score"]) for x in out}
    # some models return LABEL_0/1/2
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
    ap.add_argument("--model", default="roberta-large-mnli", help="NLI model name")
    args = ap.parse_args()

    # AUTO device: GPU if available else CPU
    device = 0 if torch.cuda.is_available() else -1
    nli = pipeline("text-classification", model=args.model, return_all_scores=True, device=device)

    df = pd.read_csv(args.csv)

    scores = []
    preds = []

    for _, row in df.iterrows():
        r1 = "" if pd.isna(row.get("Review 1")) else str(row.get("Review 1"))
        r2 = "" if pd.isna(row.get("Review 2")) else str(row.get("Review 2"))
        f1 = "" if pd.isna(row.get("APP Features 1")) else str(row.get("APP Features 1"))
        f2 = "" if pd.isna(row.get("App Features 2")) else str(row.get("App Features 2"))

        s12 = entail_prob(nli, r1, H(f2))   # Review1 supports Feature2
        s21 = entail_prob(nli, r2, H(f1))   # Review2 supports Feature1
        score = min(s12, s21)

        scores.append(score)
        preds.append(1 if score >= args.th else 0)

    df["score"] = scores
    df["pred"] = preds
    df.to_csv(args.out, index=False)
    print("Wrote:", args.out, "| device:", ("GPU" if device == 0 else "CPU"))

if __name__ == "__main__":
    main()
