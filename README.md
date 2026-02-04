# kappa_score

A Python toolkit for calculating inter-annotator agreement metrics and performing NLI-based feature scoring for text analysis.

## Code Files

### 📊 `kappa.py`
**Goal:** Calculate how much annotators agree with each other when labeling data.

### 🤖 `NLI.py`
**Goal:** Determine if two reviews talk about the same app feature.

### 📈 `score_demo.py`
**Goal:** Process multiple review pairs and predict if they describe the same feature.

### 🔍 `Th_demo.py`
**Goal:** Find the optimal threshold for making predictions using cross-validation.

## Installation

```bash
pip install nltk transformers torch pandas scikit-learn numpy
```