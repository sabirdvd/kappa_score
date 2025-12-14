from transformers import pipeline

# NLI model
nli = pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True)

def entail(premise: str, hypothesis: str) -> float:
    out = nli(f"{premise} </s></s> {hypothesis}")[0]
    d = {x["label"].lower(): float(x["score"]) for x in out}
    # if model returns LABEL_0/1/2
    if "label_2" in d:
        return d["label_2"]  # entailment
    return d["entailment"]

def H(feature: str) -> str:
    return f"This sentence is about: {feature}."

def pair_score(review1: str, feature1: str, review2: str, feature2: str) -> float:
    s12 = entail(review1, H(feature2))  # Review1 supports Feature2?
    s21 = entail(review2, H(feature1))  # Review2 supports Feature1?
    return min(s12, s21)               # both must agree

def predict(review1, feature1, review2, feature2, th: float = 0.70):
    score = pair_score(review1, feature1, review2, feature2)
    pred = 1 if score >= th else 0
    return pred, score
