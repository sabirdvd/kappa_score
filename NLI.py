from transformers import pipeline

nli = pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True)

def entail_prob(premise, hypothesis):
    out = nli(f"{premise} </s></s> {hypothesis}")[0]
    return {x["label"].lower(): x["score"] for x in out}

A = "This app is a dream come true for list makers!"
B = "I still use the computer to list my items."

# Method 1 (bidirectional)
ab = entail_prob(A, B)
ba = entail_prob(B, A)
print("A=>B", ab, "B=>A", ba)

# Method 2 (feature hypothesis)
H = "The user is talking about creating lists of items."
aH = entail_prob(A, H)
bH = entail_prob(B, H)
print("A=>H", aH, "B=>H", bH)
