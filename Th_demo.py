
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from transformers import pipeline

CSV_PATH = "Sheet1.csv"

# 1) Load + clean labels
df = pd.read_csv(CSV_PATH)
df = df[df["Annotation"].isin(["0", "1"])].copy()
df["y"] = df["Annotation"].astype(int)

# 2) NLI model
nli = pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True)

def entail_prob(premise: str, hypothesis: str) -> float:
    out = nli(f"{premise} </s></s> {hypothesis}")[0]
    d = {x["label"].lower(): float(x["score"]) for x in out}
    # handle LABEL_0/1/2 case
    if "label_2" in d:
        return d["label_2"]  # entailment
    return d["entailment"]

def H(feature: str) -> str:
    return f"This sentence is about: {feature}."

# 3) Build one score per row (cross entailment)
scores = []
for _, r in df.iterrows():
    r1 = str(r["Review 1"])
    r2 = str(r["Review 2"])
    f1 = str(r["APP Features 1"])
    f2 = str(r["App Features 2"])

    s12 = entail_prob(r1, H(f2))   # does Review1 support Feature2?
    s21 = entail_prob(r2, H(f1))   # does Review2 support Feature1?
    scores.append(min(s12, s21))   # both must agree

scores = np.array(scores)
y = df["y"].to_numpy()

# 4) Pick threshold t using 5-fold CV (max F1)
def best_threshold(s_tr, y_tr):
    best_t, best_f1 = 0.0, -1.0
    for t in np.unique(s_tr):
        pred = (s_tr >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_tr, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ts, rows = [], []

for tr, te in skf.split(scores, y):
    t = best_threshold(scores[tr], y[tr])
    ts.append(t)

    pred = (scores[te] >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y[te], pred, average="binary", zero_division=0)
    rows.append((t, p, r, f1))

print("CV thresholds:", [round(x, 3) for x in ts])
print("Final threshold (median):", round(float(np.median(ts)), 3))
print("CV avg (precision, recall, f1):",
      tuple(round(float(np.mean([x[i] for x in rows])), 3) for i in [1,2,3]))

# 5) Predict function for new pairs
FINAL_T = float(np.median(ts))

def predict_same_feature(review1, feature1, review2, feature2, t=FINAL_T):
    s12 = entail_prob(review1, H(feature2))
    s21 = entail_prob(review2, H(feature1))
    score = min(s12, s21)
    ans = 1 if score >= t else 0
    return ans, score

# Example:
# ans, score = predict_same_feature("...", "list makers", "...", "list my items")
# print(ans, score)
