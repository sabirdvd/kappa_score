# from transformers import pipeline

# nli = pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True)

# def entail_prob(premise, hypothesis):
#     out = nli(f"{premise} </s></s> {hypothesis}")[0]
#     return {x["label"].lower(): x["score"] for x in out}

# A = "This app is a dream come true for list makers!"
# B = "I still use the computer to list my items."

# # Method 1 (bidirectional)
# ab = entail_prob(A, B)
# ba = entail_prob(B, A)
# print("A=>B", ab, "B=>A", ba)

# # Method 2 (feature hypothesis)
# H = "The user is talking about creating lists of items."
# aH = entail_prob(A, H)
# bH = entail_prob(B, H)
# print("A=>H", aH, "B=>H", bH)

from transformers import pipeline

nli = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    return_all_scores=True
)

def entailment_score(premise: str, hypothesis: str) -> float:
    out = nli(f"{premise} </s></s> {hypothesis}")[0]
    scores = {x["label"].lower(): x["score"] for x in out}
    return scores["entailment"]

def make_hyp(feature: str) -> str:
    return f"This sentence is about: {feature}."

def feature_equivalence(R1, F1, R2, F2, th=0.7):
    H1 = make_hyp(F1)
    H2 = make_hyp(F2)

    e12 = entailment_score(R1, H2)  # R1 => H2
    e21 = entailment_score(R2, H1)  # R2 => H1

    score = min(e12, e21)
    pred = int(score >= th)

    return {
        "H1": H1,
        "H2": H2,
        "e12": e12,
        "e21": e21,
        "score": score,
        "pred": pred
    }

# Example
F1 = "list makers"
R1 = "This app is a dream come true for list makers!"
F2 = "list my items"
R2 = "...use the computer to list my items."

print(feature_equivalence(R1, F1, R2, F2, th=0.7))
