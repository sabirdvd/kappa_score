# kappa_score

A Python toolkit for calculating inter-annotator agreement metrics and performing NLI-based feature scoring for text analysis.

## Installation

```bash
pip install nltk transformers torch pandas scikit-learn numpy
```

## Code Files

### `kappa.py`
**Goal:** Calculate how much annotators agree with each other when labeling data.

### `NLI.py`
**Goal:** Determine if two reviews talk about the same app feature.

###  `score_demo.py`
**Goal:** Process multiple review pairs and predict if they describe the same feature.

###  `Th_demo.py`
**Goal:** Find the optimal threshold for making predictions using cross-validation.

### `nli_enhanced_eval.py` 

| # | System Variant | Best th | Acc | F1 | Kappa | BalAcc | TP | TN | FP | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | NLI only | 0.49 | 0.6667 | 0.6905 | 0.3504 | 0.6817 | 29 | 23 | 20 | 6 |
| 2 | Similarity only | 0.73 | 0.6667 | 0.7111 | 0.3603 | 0.6897 | 32 | 20 | 23 | 3 |
| 3 | Combined (alpha=0.50) | 0.67 | 0.6795 | 0.6835 | 0.3673 | 0.6880 | 27 | 26 | 17 | 8 |
| 4 | + Contradiction guard (tau_c=0.60) | 0.67 | 0.6795 | 0.6835 | 0.3673 | 0.6880 | 27 | 26 | 17 | 8 |
| 5 | + Rule penalty (lambda=0.30) [Final] | 0.67 | 0.7436 | 0.7297 | 0.4872 | 0.7462 | 27 | 31 | 12 | 8 |

## Takeaway

- The final pipeline (`+ rule penalty`) gives the best overall agreement: **Acc=0.7436**, **Kappa=0.4872**, **F1=0.7297**.
- Main gain comes from reducing false positives (`FP` from 17 to 12) while keeping true positives stable (`TP=27`).
- `NLI-only` and `Similarity-only` perform similarly, but combining signals plus rules gives a clear improvement.

## Example Cases

### Correct Match (High Confidence)
- Feature pair: `older movies` vs `old movies`
- Score: `0.9574` | Pred: `1` | Human: `1`
- Review1: `To many older movies all the time.`
- Review2: `Same old movies.`

### Model Error (False Negative)
- Feature pair: `likes` vs `liking`
- Score: `0.0000` | Pred: `0` | Human: `1`
- Review1 (short): `...video calling n likes & comments features...`
- Review2 (short): `...Save instead of liking.`
- Interpretation: contradiction/rule penalties likely suppressed an otherwise related pair.
