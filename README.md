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

### `score_demo.py`
**Goal:** Process multiple review pairs and predict if they describe the same feature.

### `Th_demo.py`
**Goal:** Find the optimal threshold for making predictions using cross-validation.

---

## `nli_enhanced_eval.py` — NLI + Similarity Pipeline with Threshold Tuning & Optional LLM Judge

### Overview

This script is a **semantic matching evaluation pipeline** that determines whether two feature-review pairs describe the **same meaning/functionality**. It combines three signals: NLI (Natural Language Inference), similarity scoring, and optionally an LLM judge.

```
CSV Input
    ↓
Resolve Labels (consensus / majority / individual)
    ↓
NLI Scoring  +  Similarity Scoring (Jaccard / Cosine)
    ↓
Grid Search over Hyperparameters (CV-tuned threshold)
    ↓
Best Config → Triage (auto_positive / needs_review / auto_negative)
    ↓
[Optional] LLM Judge on uncertain cases
    ↓
Ablation Table + Output CSVs
```

---

### Input Data

The CSV must contain these columns:

| Column | Description |
|---|---|
| `APP Features 1` | Feature description for pair A |
| `Review 1` | User review for pair A |
| `App Features 2` | Feature description for pair B |
| `Review 2` | User review for pair B |
| `Fiaz` | Annotator 1 binary label (0/1) |
| `Naveen` | Annotator 2 binary label (0/1) |

---

### Pipeline Components

#### 1. NLI Scoring
Uses a **cross-encoder NLI model** (e.g., `roberta-large-mnli`, `DeBERTa-v3-base-mnli-fever-anli`) to check if reviews entail feature hypotheses:

```
Review 1 → [NLI model] → Hypothesis(Feature 2)   → e12, c12
Review 2 → [NLI model] → Hypothesis(Feature 1)   → e21, c21
```

- Templates transform features into natural hypotheses (e.g., `"The app can {feature}."`)
- Two directions are scored and then **aggregated**

**NLI Score Modes:**
- `contra_norm`: `e / (e + c)` — normalizes entailment against contradiction
- `raw`: uses raw entailment probability

#### 2. Similarity Scoring
Measures surface/semantic overlap between pairs:
- **Jaccard**: Token overlap between features and reviews
- **Cosine**: Embedding similarity using a sentence transformer
- **Blend**: `β * cosine + (1 - β) * jaccard`

#### 3. Score Fusion

```
blended = α * nli_score + (1 - α) * similarity_score
guarded = blended  [set to 0 if contradiction ≥ τ_c]
final   = guarded * (1 - λ * rule_flag)
```

#### 4. Triage System
Each row gets a label based on thresholds `[low_th, high_th]`:

| Score Range | Label |
|---|---|
| `≥ high_th` | `auto_positive` |
| `≤ low_th` | `auto_negative` |
| In between | `needs_review` |

#### 5. LLM Judge (Optional)
Applied to `needs_review` (or all) rows. Calls an OpenAI-compatible API and returns:
```json
{"label": "same|different", "confidence": 0.0–1.0, "rationale": "..."}
```
Supports **self-consistency voting** (`--llm-votes`) and **in-context learning** (`--llm-icl-shots`).

---

### Usage Examples

**Run without LLM:**
```bash
python3 nli_enhanced_eval.py \
  --csv "Ground Truth.csv" \
  --model roberta-large-mnli \
  --target-label consensus_only \
  --objective kappa \
  --cv-folds 5 \
  --similarity-method cosine \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --nli-score-mode contra_norm
```

**Run with GPT-5 API judge:**
```bash
python3 nli_enhanced_eval.py \
  --csv "Ground Truth.csv" \
  --model roberta-large-mnli \
  --target-label consensus_only \
  --objective kappa \
  --cv-folds 5 \
  --similarity-method cosine \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --nli-score-mode contra_norm \
  --llm-judge --llm-gpt5-api --llm-model gpt-5
```

**Best result with GPT-5 and DeBERTa-v3-ANLI:**
```bash
python3 nli_enhanced_eval.py \
  --csv "Ground Truth.csv" \
  --model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
  --target-label consensus_only \
  --objective kappa \
  --cv-folds 5 \
  --similarity-method cosine \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --nli-score-mode contra_norm \
  --llm-judge --llm-gpt5-api --llm-model gpt-5 \
  --llm-on needs_review --llm-confidence-th 0.6 --llm-votes 3
```

---

### Hyperparameters

#### NLI Model
| Argument | Default | Description |
|---|---|---|
| `--model` | `roberta-large-mnli` | HuggingFace NLI model |
| `--nli-score-mode` | `contra_norm` | `contra_norm` or `raw` entailment scoring |

#### Similarity
| Argument | Default | Description |
|---|---|---|
| `--similarity-method` | `blend` | `jaccard`, `cosine`, or `blend` |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence transformer for cosine similarity |
| `--similarity-beta` | `0.7` | Blend weight: `β*cosine + (1-β)*jaccard` |

#### Grid Search (tuned via CV)
| Argument | Default | Description |
|---|---|---|
| `--alphas` | `0.5,0.7,0.9,1.0` | Weight of NLI vs similarity (`α`) |
| `--contradiction-thresholds` | `0.6,0.7,0.8,0.9,1.01` | Threshold τ_c to zero-out contradictory pairs |
| `--rule-penalties` | `0.0,0.15,0.30` | Penalty λ for "copied review, divergent feature" rule |
| `--templates` | `all` | Which NLI hypothesis templates to try |
| `--aggregators` | `all` | How to combine bidirectional NLI scores (`min`, `mean`, `geometric`, `harmonic`) |

#### Cross-Validation
| Argument | Default | Description |
|---|---|---|
| `--cv-folds` | `5` | Number of stratified K-folds |
| `--objective` | `kappa` | Metric to optimise: `kappa`, `f1`, or `balanced_acc` |
| `--seed` | `42` | Random seed for fold splits |

#### Triage Thresholds
| Argument | Default | Description |
|---|---|---|
| `--min-pos-precision` | `0.90` | Min precision for `auto_positive` boundary |
| `--min-neg-precision` | `0.90` | Min NPV for `auto_negative` boundary |

#### LLM Judge
| Argument | Default | Description |
|---|---|---|
| `--llm-judge` | `False` | Enable LLM second-stage |
| `--llm-model` | `gpt-5` | Model to use (`gpt-5`, `gpt-5-mini`, `gpt-oss-120b`, `gpt-oss-20b`) |
| `--llm-confidence-th` | `0.70` | Min confidence to override base prediction |
| `--llm-votes` | `1` | Self-consistency votes (majority wins) |
| `--llm-uncertainty-band` | `0.05` | Only judge rows where `\|score - threshold\| ≤ band` |
| `--llm-on` | `needs_review` | Apply to `needs_review` or `all` rows |
| `--llm-icl-shots` | `0` | In-context examples in prompt (`0` or `3`) |
| `--llm-temperature` | `0.0` | Sampling temperature |
| `--llm-require-unanimous` | `False` | Require 100% vote agreement to override |

---

### Ablation Results (roberta-large-mnli)

| # | Model | Best th | Acc | F1 | Kappa | BalAcc | TP | TN | FP | FN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | NLI only | 0.49 | 0.6667 | 0.6905 | 0.3504 | 0.6817 | 29 | 23 | 20 | 6 |
| 2 | Similarity only | 0.73 | 0.6667 | 0.7111 | 0.3603 | 0.6897 | 32 | 20 | 23 | 3 |
| 3 | Combined (alpha=0.50) | 0.67 | 0.6795 | 0.6835 | 0.3673 | 0.6880 | 27 | 26 | 17 | 8 |
| 4 | + Contradiction guard (tau_c=0.60) | 0.67 | 0.6795 | 0.6835 | 0.3673 | 0.6880 | 27 | 26 | 17 | 8 |
| 5 | + Rule penalty (lambda=0.30) [Final] | 0.67 | 0.7436 | 0.7297 | 0.4872 | 0.7462 | 27 | 31 | 12 | 8 |

---

### Takeaway

- The final pipeline (`+ rule penalty`) gives the best overall agreement: **Acc=0.7436**, **Kappa=0.4872**, **F1=0.7297**.
- Main gain comes from reducing false positives (`FP` from 17 → 12) while keeping true positives stable (`TP=27`).
- `NLI-only` and `Similarity-only` perform similarly, but combining signals plus rules gives a clear improvement.
- Best backbone: **DeBERTa-v3-base-mnli-fever-anli** with GPT-5 judge and 3-vote majority on uncertain cases.

---

### Example Cases

#### Correct Match (High Confidence)
- Feature pair: `older movies` vs `old movies`
- Score: `0.9574` | Pred: `1` | Human: `1`
- Review1: `To many older movies all the time.`
- Review2: `Same old movies.`

#### Model Error (False Negative)
- Feature pair: `likes` vs `liking`
- Score: `0.0000` | Pred: `0` | Human: `1`
- Review1: `...video calling n likes & comments features...`
- Review2: `...Save instead of liking.`
- Interpretation: contradiction/rule penalties likely suppressed an otherwise related pair.

---

### Output Files

| File | Description |
|---|---|
| `enhanced_config_search.csv` | All grid search configs ranked by objective |
| `enhanced_row_scores.csv` | Per-row scores, predictions, triage labels, and LLM results |
| `enhanced_triage.csv` | Decision-routing file for downstream use |
| `process_ablation_table.csv` | Ablation table (CSV) |
| `process_ablation_table.md` | Ablation table (Markdown) |

### The `enhanced_triage.csv` Decision-Routing File

It is not just a 0/1 prediction. It assigns each pair to one of three actions:

- `auto_positive` → high-confidence match (can auto-accept as 1)
- `auto_negative` → high-confidence non-match (can auto-accept as 0)
- `needs_review` → uncertain case, route to human (or LLM judge)