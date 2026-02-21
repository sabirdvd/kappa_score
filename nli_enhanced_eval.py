#!/usr/bin/env python3
"""NLI + similarity matcher with threshold tuning and optional LLM as a judge.

Run without LLM:
python3 nli_enhanced_eval.py \
  --csv "Ground Truth.csv" \
  --model roberta-large-mnli \
  --target-label consensus_only \
  --objective kappa \
  --cv-folds 5 \
  --similarity-method cosine \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --nli-score-mode contra_norm

Run with GPT-5 API judge:
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

Run with GPT-5-mini API judge:
python3 nli_enhanced_eval.py \
  --csv "Ground Truth.csv" \
  --model roberta-large-mnli \
  --target-label consensus_only \
  --objective kappa \
  --cv-folds 5 \
  --similarity-method cosine \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --nli-score-mode contra_norm \
  --llm-judge --llm-gpt5-api --llm-model gpt-5-mini
  

Best result with GPT-5 and DeBERTa-v3_ANLI
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
  
"""

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

COL_F1 = "APP Features 1"
COL_R1 = "Review 1"
COL_F2 = "App Features 2"
COL_R2 = "Review 2"
COL_FIAZ = "Fiaz"
COL_NAVEEN = "Naveen"
REQUIRED_COLUMNS = [COL_F1, COL_R1, COL_F2, COL_R2, COL_FIAZ, COL_NAVEEN]


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def jaccard(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    cos = float(np.dot(a, b) / denom)
    # Map cosine [-1,1] -> [0,1] for easy fusion with probability-like scores.
    return max(0.0, min(1.0, 0.5 * (cos + 1.0)))


def majority_or(fiaz: pd.Series, naveen: pd.Series) -> pd.Series:
    f = safe_int_series(fiaz)
    n = safe_int_series(naveen)
    return ((f + n) >= 1).astype(int)


def consensus_mask(fiaz: pd.Series, naveen: pd.Series) -> pd.Series:
    f = safe_int_series(fiaz)
    n = safe_int_series(naveen)
    return f == n


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    po = float((y_true == y_pred).sum()) / len(y_true)
    p1 = float((y_true == 1).sum()) / len(y_true)
    p2 = float((y_pred == 1).sum()) / len(y_true)
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = (recall + tnr) / 2.0
    kappa = cohen_kappa(y_true, y_pred)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "balanced_acc": bal_acc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def stratified_kfold_indices(y: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, val in enumerate(idx):
            folds[i % n_splits].append(int(val))
    out = []
    for f in folds:
        arr = np.array(f, dtype=int)
        rng.shuffle(arr)
        out.append(arr)
    return out


def normalize_feature_text(text: str) -> str:
    t = str(text).strip().strip('"').strip()
    return t.rstrip(".!? ")


def find_label_indices(model) -> tuple[int, int]:
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    entail_idx = -1
    contra_idx = -1
    for idx, label in id2label.items():
        if "entail" in label:
            entail_idx = idx
        if "contrad" in label:
            contra_idx = idx
    if entail_idx < 0 or contra_idx < 0:
        raise ValueError(f"Could not find entailment/contradiction labels in {model.config.id2label}")
    return entail_idx, contra_idx


def nli_probs(premise: str, hypothesis: str, tokenizer, model, device: str) -> np.ndarray:
    enc = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    return probs


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def build_embeddings(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    uniq = list(dict.fromkeys(texts))
    out: Dict[str, np.ndarray] = {}
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            model_out = model(**enc)
            emb = mean_pooling(model_out.last_hidden_state, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb_np = emb.detach().cpu().numpy()
        for t, v in zip(batch, emb_np):
            out[t] = v
    return out


def get_templates() -> Dict[str, str]:
    return {
        "about": "This sentence is about: {feature}.",
        "can_do": "The review says the app can {feature}.",
        "functionality": "The user is describing {feature} functionality.",
        "mentions": "The review mentions the feature: {feature}.",
        "user_wants": "The user talks about wanting to {feature}.",
    }


def get_aggregators() -> Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    eps = 1e-12
    return {
        "min": lambda a, b: np.minimum(a, b),
        "mean": lambda a, b: (a + b) / 2.0,
        "geometric": lambda a, b: np.sqrt(np.maximum(a * b, eps)),
        "harmonic": lambda a, b: (2.0 * a * b) / np.maximum(a + b, eps),
    }


def normalize_entail_with_contradiction(e: np.ndarray, c: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return e / (e + c + eps)


def model_result_dir_name(model_name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    return f"{clean}_result"


def objective_value(m: Dict[str, float], objective: str) -> float:
    if objective == "f1":
        return m["f1"]
    if objective == "balanced_acc":
        return m["balanced_acc"]
    return m["kappa"]


def best_threshold_and_metrics_by_kappa(score: np.ndarray, y: np.ndarray) -> tuple[float, Dict[str, float]]:
    ths = np.linspace(0.01, 0.99, 99)
    best_th = 0.5
    best_m: Optional[Dict[str, float]] = None
    for th in ths:
        pred = (score >= th).astype(int)
        m = metrics(y, pred)
        if best_m is None or (m["kappa"], m["f1"], m["acc"]) > (best_m["kappa"], best_m["f1"], best_m["acc"]):
            best_m = m
            best_th = float(th)
    assert best_m is not None
    return best_th, best_m


@dataclass
class Config:
    template: str
    aggregator: str
    alpha: float
    contradiction_th: float
    rule_penalty: float
    tuned_th: float
    cv_acc: float
    cv_f1: float
    cv_kappa: float
    cv_balanced_acc: float


def tune_threshold_cv(
    y: np.ndarray,
    score: np.ndarray,
    folds: List[np.ndarray],
    objective: str,
    threshold_grid: np.ndarray,
) -> tuple[float, Dict[str, float]]:
    best_th = 0.5
    best_m: Optional[Dict[str, float]] = None
    best_obj = -1e9
    for th in threshold_grid:
        fold_ms = []
        for fold in folds:
            p = (score[fold] >= th).astype(int)
            fold_ms.append(metrics(y[fold], p))
        avg = {
            "acc": float(np.mean([m["acc"] for m in fold_ms])),
            "f1": float(np.mean([m["f1"] for m in fold_ms])),
            "kappa": float(np.mean([m["kappa"] for m in fold_ms])),
            "balanced_acc": float(np.mean([m["balanced_acc"] for m in fold_ms])),
        }
        obj = objective_value(avg, objective)
        if best_m is None or (obj, avg["f1"], avg["acc"]) > (best_obj, best_m["f1"], best_m["acc"]):
            best_obj = obj
            best_m = avg
            best_th = float(th)
    assert best_m is not None
    return best_th, best_m


def find_triage_thresholds(
    y: np.ndarray,
    score: np.ndarray,
    min_pos_precision: float,
    min_neg_precision: float,
) -> tuple[float, float]:
    ths = np.linspace(0.01, 0.99, 99)
    high = 0.99
    low = 0.01
    for th in ths:
        p = (score >= th).astype(int)
        m = metrics(y, p)
        if m["precision"] >= min_pos_precision:
            high = float(th)
            break
    for th in ths:
        p = (score > th).astype(int)
        tn = ((p == 0) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        if npv >= min_neg_precision:
            low = float(th)
    if low > high:
        low, high = high, low
    return low, high


def _extract_response_text(resp: Dict) -> str:
    if isinstance(resp.get("output_text"), str) and resp["output_text"]:
        return resp["output_text"]
    # Some structured-output responses may expose parsed JSON directly.
    if isinstance(resp.get("output_parsed"), (dict, list)):
        try:
            return json.dumps(resp["output_parsed"])
        except Exception:
            pass
    out = resp.get("output", [])
    texts: List[str] = []
    for item in out:
        for c in item.get("content", []):
            if c.get("type") in {"output_text", "text"} and c.get("text"):
                texts.append(c["text"])
            elif c.get("type") in {"output_json", "json"} and c.get("json") is not None:
                try:
                    texts.append(json.dumps(c["json"]))
                except Exception:
                    pass
    return "\n".join(texts)


def _parse_judge_json(text: str) -> tuple[Optional[int], float, str]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    raw = m.group(0) if m else text
    try:
        obj = json.loads(raw)
    except Exception:
        obj = None
    if obj is None:
        # Fallback for non-JSON local model outputs.
        t = text.strip().lower()
        lbl: Optional[int] = None
        if re.search(r"\b(same|match|equivalent|yes)\b", t):
            lbl = 1
        if re.search(r"\b(different|mismatch|not\s+same|no)\b", t):
            # Prefer explicit negative if both patterns appear.
            lbl = 0
        conf = 0.0
        m_conf = re.search(r"(confidence|conf)\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)", t)
        if m_conf:
            try:
                conf = float(m_conf.group(2))
            except Exception:
                conf = 0.0
        return lbl, max(0.0, min(1.0, conf)), text[:300]
    label = obj.get("label")
    conf = obj.get("confidence", 0.0)
    rationale = str(obj.get("rationale", ""))[:300]
    lbl: Optional[int] = None
    if isinstance(label, str):
        t = label.strip().lower()
        if t in {"same", "1", "yes", "match"}:
            lbl = 1
        elif t in {"different", "0", "no", "mismatch"}:
            lbl = 0
    elif isinstance(label, (int, float)):
        lbl = int(label >= 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return lbl, conf, rationale


def build_llm_user_prompt(f1: str, r1: str, f2: str, r2: str, icl_shots: int) -> str:
    if icl_shots == 3:
        shots = (
            "Example 1\n"
            "Pair A:\nFeature: dark mode\nReview: I love the dark theme at night.\n"
            "Pair B:\nFeature: dark mode\nReview: The app finally has night mode.\n"
            'Answer: {"label":"same","confidence":0.95,"rationale":"Both describe dark mode."}\n\n'
            "Example 2\n"
            "Pair A:\nFeature: export pdf\nReview: I can save invoices as PDF.\n"
            "Pair B:\nFeature: cloud backup\nReview: My data syncs to cloud automatically.\n"
            'Answer: {"label":"different","confidence":0.97,"rationale":"Different product functions."}\n\n'
            "Example 3\n"
            "Pair A:\nFeature: add items to list\nReview: I can quickly list my groceries.\n"
            "Pair B:\nFeature: list maker\nReview: This app is perfect for making lists.\n"
            'Answer: {"label":"same","confidence":0.88,"rationale":"Same list-creation intent."}\n\n'
        )
        query = (
            f"Now judge this case.\n"
            f"Pair A:\nFeature: {f1}\nReview: {r1}\n\n"
            f"Pair B:\nFeature: {f2}\nReview: {r2}\n\n"
            "Answer JSON only."
        )
        return shots + query
    return (
        f"Pair A:\nFeature: {f1}\nReview: {r1}\n\n"
        f"Pair B:\nFeature: {f2}\nReview: {r2}\n\n"
        "Are Pair A and Pair B the same meaning?"
    )


def llm_judge_once(
    api_base: str,
    api_key: str,
    model: str,
    f1: str,
    r1: str,
    f2: str,
    r2: str,
    icl_shots: int,
    temperature: float,
    max_output_tokens: int,
    timeout_sec: int,
) -> tuple[Optional[int], float, str]:
    sys = (
        "You are a strict semantic judge. Compare two feature-review pairs. "
        "Return JSON only: "
        "{\"label\":\"same|different\",\"confidence\":0..1,\"rationale\":\"short\"}. "
        "Keep rationale under 15 words."
    )
    usr = build_llm_user_prompt(f1, r1, f2, r2, icl_shots)
    model_l = str(model).lower()
    # Use the simpler Responses API format for GPT-5 family.
    if model_l.startswith("gpt-5"):
        payload = {
            "model": model,
            "instructions": sys,
            "input": usr,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": "low"},
        }
    else:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
            "max_output_tokens": max_output_tokens,
        }
    # GPT-5 family does not support temperature on Responses API.
    # For other models/endpoints, send it only when explicitly > 0.
    supports_temperature = not (model_l.startswith("gpt-5"))
    if supports_temperature and temperature is not None and float(temperature) > 0.0:
        payload["temperature"] = float(temperature)
    # Force strict machine-readable output for GPT-5 family.
    if model_l.startswith("gpt-5"):
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "semantic_judge",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string", "enum": ["same", "different"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string", "maxLength": 120},
                    },
                    "required": ["label", "confidence", "rationale"],
                },
            }
        }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _post(p: Dict) -> Dict:
        url = api_base.rstrip("/") + "/responses"
        req = urllib.request.Request(
            url=url,
            data=json.dumps(p).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    try:
        req_payload = dict(payload)
        for _attempt in range(3):
            obj = _post(req_payload)
            if isinstance(obj, dict) and obj.get("error"):
                try:
                    emsg = json.dumps(obj["error"])[:300]
                except Exception:
                    emsg = str(obj.get("error"))[:300]
                return None, 0.0, f"API error: {emsg}"
            text = _extract_response_text(obj)
            if text:
                return _parse_judge_json(text)

            status = str(obj.get("status", "")).lower() if isinstance(obj, dict) else ""
            details = obj.get("incomplete_details") if isinstance(obj, dict) else {}
            reason = str((details or {}).get("reason", "")).lower()
            if status == "incomplete" and ("max_output" in reason or "max_tokens" in reason):
                prev = int(req_payload.get("max_output_tokens", max_output_tokens))
                req_payload["max_output_tokens"] = min(max(prev * 2, prev + 128), 2048)
                continue
            return None, 0.0, f"Empty model output: {str(obj)[:300]}"
        return None, 0.0, f"Incomplete after retries: {str(obj)[:300]}"
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        # Retry once without temperature if endpoint rejects it.
        if "temperature" in payload and "Unsupported parameter" in body and "temperature" in body:
            try:
                payload_retry = dict(payload)
                payload_retry.pop("temperature", None)
                obj = _post(payload_retry)
                text = _extract_response_text(obj)
                return _parse_judge_json(text)
            except Exception:
                pass
        msg = f"HTTP {e.code} {e.reason}: {body[:300]}"
        return None, 0.0, msg
    except Exception as e:
        return None, 0.0, f"{type(e).__name__}: {str(e)[:300]}"


def llm_judge_vote(
    api_base: str,
    api_key: str,
    model: str,
    f1: str,
    r1: str,
    f2: str,
    r2: str,
    icl_shots: int,
    temperature: float,
    max_output_tokens: int,
    timeout_sec: int,
    votes: int,
) -> tuple[Optional[int], float, str, float]:
    labels: List[int] = []
    confs: List[float] = []
    rationales: List[str] = []
    errors: List[str] = []
    for _ in range(max(1, votes)):
        lbl, conf, rat = llm_judge_once(
            api_base, api_key, model, f1, r1, f2, r2, icl_shots, temperature, max_output_tokens, timeout_sec
        )
        if lbl is not None:
            labels.append(lbl)
            confs.append(conf)
            if rat:
                rationales.append(rat)
        elif rat:
            errors.append(rat)
    if not labels:
        return None, 0.0, (errors[0] if errors else ""), 0.0
    ones = sum(labels)
    zeros = len(labels) - ones
    final = 1 if ones >= zeros else 0
    conf = float(np.mean(confs)) if confs else 0.0
    rat = rationales[0] if rationales else ""
    agreement = float(max(ones, zeros) / max(1, len(labels)))
    return final, conf, rat, agreement


def llm_server_reachable(api_base: str, timeout_sec: int) -> bool:
    base = api_base.rstrip("/")
    probes = [base + "/models", base + "/health", base + "/v1/models", base + "/v1/health"]
    for url in probes:
        req = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                if 200 <= int(resp.status) < 500:
                    return True
        except Exception:
            continue
    return False


def parse_float_list(raw: str, arg_name: str) -> List[float]:
    vals: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError as exc:
            raise ValueError(f"Invalid float in {arg_name}: {part}") from exc
    return vals


def select_named_variants(
    all_items: Dict[str, object],
    selected: str,
    arg_name: str,
) -> Dict[str, object]:
    if selected == "all":
        return all_items
    keys = [x.strip() for x in selected.split(",") if x.strip()]
    chosen = {k: all_items[k] for k in keys if k in all_items}
    if not chosen:
        raise ValueError(f"No valid {arg_name} selected from: {keys}")
    return chosen


def apply_llm_mode_presets(args: argparse.Namespace) -> None:
    if args.llm_gpt5_api and args.llm_open_source:
        raise ValueError("Use only one of --llm-gpt5-api or --llm-open-source.")
    if args.llm_gpt5_api:
        args.llm_judge = True
        if args.llm_model in {"gpt-oss-120b", "gpt-oss-20b"}:
            args.llm_model = "gpt-5"
        args.llm_api_base = "https://api.openai.com/v1"
        args.llm_api_key_env = "OPENAI_API_KEY"
        # Safer default policy for GPT judges: only uncertain rows, high confidence.
        if float(getattr(args, "llm_confidence_th", 0.70)) < 0.90:
            args.llm_confidence_th = 0.90
    if args.llm_open_source:
        args.llm_judge = True
        if args.llm_model in {"gpt-5", "gpt-5-mini"}:
            args.llm_model = "gpt-oss-20b"
        if args.llm_api_key_env == "OPENAI_API_KEY":
            args.llm_api_key_env = "NONE"


def resolve_target_labels(df: pd.DataFrame, target_label: str) -> tuple[pd.DataFrame, np.ndarray]:
    fiaz = safe_int_series(df[COL_FIAZ])
    naveen = safe_int_series(df[COL_NAVEEN])
    consensus = consensus_mask(fiaz, naveen)

    if target_label == "fiaz":
        y_full = fiaz.to_numpy()
        eval_idx = np.arange(len(df))
    elif target_label == "naveen":
        y_full = naveen.to_numpy()
        eval_idx = np.arange(len(df))
    elif target_label == "majority_or":
        y_full = majority_or(fiaz, naveen).to_numpy()
        eval_idx = np.arange(len(df))
    else:
        y_full = fiaz[consensus].to_numpy()
        eval_idx = np.where(consensus.values)[0]

    eval_df = df.iloc[eval_idx].copy().reset_index(drop=True)
    return eval_df, y_full.astype(int)


def run_llm_judge_stage(row_df: pd.DataFrame, args: argparse.Namespace, tuned_th: float) -> pd.DataFrame:
    api_key = ""
    if args.llm_api_key_env.upper() != "NONE":
        api_key = os.environ.get(args.llm_api_key_env, "").strip()
    is_local_base = args.llm_api_base.startswith("http://localhost") or args.llm_api_base.startswith(
        "http://127.0.0.1"
    )
    if is_local_base and not llm_server_reachable(args.llm_api_base, args.llm_timeout_sec):
        raise RuntimeError(
            "Local LLM endpoint is not reachable. Start an OpenAI-compatible server first "
            f"(api_base={args.llm_api_base})."
        )
    if not api_key and not is_local_base:
        print(
            f"LLM judge requested, but env var {args.llm_api_key_env} is empty and api-base is non-local. "
            "Skipping LLM stage."
        )
        return row_df

    if args.llm_on == "needs_review":
        idx = row_df.index[row_df["triage_label"] == "needs_review"].tolist()
    else:
        idx = row_df.index.tolist()
    if args.llm_uncertainty_band > 0:
        band = float(args.llm_uncertainty_band)
        idx = [i for i in idx if abs(float(row_df.at[i, "final_score"]) - float(tuned_th)) <= band]
    if args.llm_max_cases > 0:
        idx = idx[: args.llm_max_cases]

    overrides = 0
    judged = 0
    failed = 0
    eligible = len(idx)
    failure_reasons: List[str] = []
    for i in idx:
        r = row_df.loc[i]
        lbl, conf, rat, agree = llm_judge_vote(
            api_base=args.llm_api_base,
            api_key=api_key,
            model=args.llm_model,
            f1=str(r[COL_F1]),
            r1=str(r[COL_R1]),
            f2=str(r[COL_F2]),
            r2=str(r[COL_R2]),
            icl_shots=int(args.llm_icl_shots),
            temperature=float(args.llm_temperature),
            max_output_tokens=int(args.llm_max_output_tokens),
            timeout_sec=int(args.llm_timeout_sec),
            votes=int(args.llm_votes),
        )
        if lbl is None:
            failed += 1
            if rat:
                failure_reasons.append(rat)
            continue
        judged += 1
        row_df.at[i, "llm_label"] = int(lbl)
        row_df.at[i, "llm_confidence"] = float(conf)
        row_df.at[i, "llm_rationale"] = rat
        unanimous_ok = (not args.llm_require_unanimous) or (agree >= 0.999999)
        if conf >= args.llm_confidence_th and unanimous_ok:
            base_pred = int(row_df.at[i, "pred_final"])
            row_df.at[i, "pred_final"] = int(lbl)
            if base_pred != int(lbl):
                overrides += 1
                row_df.at[i, "llm_override"] = 1
            row_df.at[i, "triage_label"] = "auto_positive_llm" if int(lbl) == 1 else "auto_negative_llm"

    print(
        f"LLM judge(api): model={args.llm_model} judged={judged} overrides={overrides} "
        f"failed={failed} (on={args.llm_on}, confidence_th={args.llm_confidence_th:.2f}, "
        f"band={args.llm_uncertainty_band:.3f}, unanimous={args.llm_require_unanimous})"
    )
    if failed > 0 and failure_reasons:
        # Print only a few unique reasons to keep terminal output readable.
        seen = []
        for msg in failure_reasons:
            if msg not in seen:
                seen.append(msg)
            if len(seen) >= 3:
                break
        for i, msg in enumerate(seen, start=1):
            print(f"LLM error {i}: {msg}")
    if args.llm_fail_on_error and failed > 0:
        raise RuntimeError(
            f"LLM judge had {failed} failed calls out of {eligible} eligible rows "
            f"(judged={judged}, overrides={overrides})."
        )
    if eligible > 0 and judged == 0:
        msg = (
            "LLM judge ran but zero rows were judged successfully. "
            "Check llm-api-base, model name, server health, and JSON output format."
        )
        if args.llm_strict:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
    elif judged > 0 and overrides == 0:
        print(
            "WARNING: LLM judge returned labels but made zero overrides. "
            "This can be valid, or confidence threshold may be too high."
        )
    return row_df


def ablation_markdown_lines(ablation_df: pd.DataFrame) -> List[str]:
    md_lines = []
    md_lines.append("| # | Model | Best th | Acc | F1 | Kappa | BalAcc | TP | TN | FP | FN |")
    md_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ablation_df.iterrows():
        best_th_str = "-" if pd.isna(r["Best th"]) else f"{float(r['Best th']):.2f}"
        md_lines.append(
            f"| {int(r['#'])} | {r['Model']} | {best_th_str} | {r['Acc']:.4f} | {r['F1']:.4f} | "
            f"{r['Kappa']:.4f} | {r['BalAcc']:.4f} | {int(r['TP'])} | {int(r['TN'])} | {int(r['FP'])} | {int(r['FN'])} |"
        )
    return md_lines


def print_ablation_terminal(ablation_df: pd.DataFrame) -> None:
    cols = ["#", "Model", "Best th", "Acc", "F1", "Kappa", "BalAcc", "TP", "TN", "FP", "FN"]
    rows: List[List[str]] = []
    for _, r in ablation_df.iterrows():
        rows.append(
            [
                str(int(r["#"])),
                str(r["Model"]),
                "-" if pd.isna(r["Best th"]) else f"{float(r['Best th']):.2f}",
                f"{float(r['Acc']):.4f}",
                f"{float(r['F1']):.4f}",
                f"{float(r['Kappa']):.4f}",
                f"{float(r['BalAcc']):.4f}",
                str(int(r["TP"])),
                str(int(r["TN"])),
                str(int(r["FP"])),
                str(int(r["FN"])),
            ]
        )
    widths = [len(c) for c in cols]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def fmt_row(values: List[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print("\nAblation table")
    print(fmt_row(cols))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced NLI matching with grid search and triage.")
    parser.add_argument("--csv", default="Ground Truth.csv")
    parser.add_argument("--model", default="roberta-large-mnli")
    parser.add_argument("--target-label", choices=["consensus_only", "majority_or", "fiaz", "naveen"], default="consensus_only")
    parser.add_argument("--objective", choices=["kappa", "f1", "balanced_acc"], default="kappa")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-config", default="enhanced_config_search.csv")
    parser.add_argument("--out-rows", default="enhanced_row_scores.csv")
    parser.add_argument("--out-triage", default="enhanced_triage.csv")
    parser.add_argument("--min-pos-precision", type=float, default=0.90)
    parser.add_argument("--min-neg-precision", type=float, default=0.90)
    parser.add_argument(
        "--similarity-method",
        choices=["jaccard", "cosine", "blend"],
        default="blend",
        help="Lexical similarity signal type.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence embedding model used when similarity-method includes cosine.",
    )
    parser.add_argument(
        "--similarity-beta",
        type=float,
        default=0.7,
        help="Blend weight for cosine in blend mode: beta*cos + (1-beta)*jaccard.",
    )
    parser.add_argument("--templates", default="all", help="Comma list of template keys or 'all'.")
    parser.add_argument("--aggregators", default="all", help="Comma list of aggregator keys or 'all'.")
    parser.add_argument("--alphas", default="0.5,0.7,0.9,1.0", help="Comma list of alpha values.")
    parser.add_argument(
        "--contradiction-thresholds",
        default="0.6,0.7,0.8,0.9,1.01",
        help="Comma list of contradiction guard thresholds.",
    )
    parser.add_argument("--rule-penalties", default="0.0,0.15,0.30", help="Comma list of rule penalty values.")
    parser.add_argument(
        "--nli-score-mode",
        choices=["contra_norm", "raw"],
        default="contra_norm",
        help="How to build per-direction NLI score before aggregation.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: <hf_model_name>_result",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM second-stage judge.")
    parser.add_argument(
        "--llm-gpt5-api",
        action="store_true",
        help="Use GPT-5 via OpenAI API (auto config).",
    )
    parser.add_argument(
        "--llm-open-source",
        action="store_true",
        help="Use open-source judge through an OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5",
        choices=["gpt-5", "gpt-5-mini", "gpt-oss-120b", "gpt-oss-20b"],
        help="LLM judge model name.",
    )
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1", help="OpenAI-compatible API base.")
    parser.add_argument(
        "--llm-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name holding API key. Use 'NONE' to disable key lookup.",
    )
    parser.add_argument(
        "--llm-on",
        default="needs_review",
        choices=["needs_review", "all"],
        help="Where to apply LLM judge.",
    )
    parser.add_argument("--llm-confidence-th", type=float, default=0.70, help="Min confidence to override base pred.")
    parser.add_argument("--llm-votes", type=int, default=1, help="Self-consistency votes per case.")
    parser.add_argument(
        "--llm-uncertainty-band",
        type=float,
        default=0.05,
        help="Only apply LLM when |final_score - tuned_threshold| <= band.",
    )
    parser.add_argument(
        "--llm-require-unanimous",
        action="store_true",
        help="Require unanimous vote agreement before LLM override.",
    )
    parser.add_argument(
        "--llm-icl-shots",
        type=int,
        default=0,
        choices=[0, 3],
        help="Use in-context examples in LLM prompt (0 or 3).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="LLM temperature (sent only when > 0 and endpoint supports it).",
    )
    parser.add_argument("--llm-max-output-tokens", type=int, default=120, help="Max tokens for LLM response.")
    parser.add_argument("--llm-timeout-sec", type=int, default=60, help="HTTP timeout for LLM calls.")
    parser.add_argument(
        "--llm-max-cases",
        type=int,
        default=0,
        help="Max number of cases to judge (0 = all eligible).",
    )
    parser.add_argument(
        "--llm-strict",
        action="store_true",
        help="Fail with explicit error if LLM judge is enabled but no rows are successfully judged.",
    )
    parser.add_argument(
        "--llm-fail-on-error",
        action="store_true",
        help="Fail if any LLM judge call fails (request/timeout/parse).",
    )
    args = parser.parse_args()
    apply_llm_mode_presets(args)
    out_dir = Path(args.out_dir) if args.out_dir else Path(model_result_dir_name(args.model))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cfg = out_dir / Path(args.out_config).name
    out_rows = out_dir / Path(args.out_rows).name
    out_triage = out_dir / Path(args.out_triage).name

    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    eval_df, y = resolve_target_labels(df, args.target_label)
    folds = stratified_kfold_indices(y, args.cv_folds, args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {args.model} on device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()
    entail_idx, contra_idx = find_label_indices(model)

    use_cosine = args.similarity_method in {"cosine", "blend"}
    emb_map: Dict[str, np.ndarray] = {}
    if use_cosine:
        print(f"Loading embedding model: {args.embedding_model} on device={device}")
        emb_tok = AutoTokenizer.from_pretrained(args.embedding_model)
        emb_model = AutoModel.from_pretrained(args.embedding_model).to(device)
        emb_model.eval()
        all_texts: List[str] = []
        for _, r in eval_df.iterrows():
            all_texts.append(str(r[COL_F1]))
            all_texts.append(str(r[COL_F2]))
            all_texts.append(str(r[COL_R1]))
            all_texts.append(str(r[COL_R2]))
        emb_map = build_embeddings(all_texts, emb_tok, emb_model, device=device)

    all_templates = get_templates()
    all_aggs = get_aggregators()
    templates = select_named_variants(all_templates, args.templates, "templates")
    aggs = select_named_variants(all_aggs, args.aggregators, "aggregators")
    alphas = parse_float_list(args.alphas, "--alphas")
    contra_thresholds = parse_float_list(args.contradiction_thresholds, "--contradiction-thresholds")
    rule_penalties = parse_float_list(args.rule_penalties, "--rule-penalties")
    if not alphas:
        raise ValueError("No valid alpha values parsed from --alphas.")
    if not contra_thresholds:
        raise ValueError("No valid values parsed from --contradiction-thresholds.")
    if not rule_penalties:
        raise ValueError("No valid values parsed from --rule-penalties.")
    threshold_grid = np.linspace(0.01, 0.99, 99)

    config_rows = []
    best_cfg: Optional[Config] = None
    best_final_obj = -1e9
    best_score_vector = None
    best_aux = {}

    for tmpl_name, tmpl in templates.items():
        print(f"Template: {tmpl_name}")
        e12 = []
        e21 = []
        c12 = []
        c21 = []
        lex = []
        lex_j = []
        lex_c = []
        rule = []
        for _, r in eval_df.iterrows():
            f1 = normalize_feature_text(r[COL_F1])
            f2 = normalize_feature_text(r[COL_F2])
            h1 = tmpl.format(feature=f1)
            h2 = tmpl.format(feature=f2)
            r1 = str(r[COL_R1])
            r2 = str(r[COL_R2])

            p12 = nli_probs(r1, h2, tokenizer, model, device)
            p21 = nli_probs(r2, h1, tokenizer, model, device)
            e12.append(float(p12[entail_idx]))
            e21.append(float(p21[entail_idx]))
            c12.append(float(p12[contra_idx]))
            c21.append(float(p21[contra_idx]))

            sim_feat_j = jaccard(f1, f2)
            sim_rev_j = jaccard(r1, r2)
            jscore = 0.5 * sim_feat_j + 0.5 * sim_rev_j

            cscore = 0.0
            if use_cosine:
                ef1 = emb_map.get(str(r[COL_F1]))
                ef2 = emb_map.get(str(r[COL_F2]))
                er1 = emb_map.get(str(r[COL_R1]))
                er2 = emb_map.get(str(r[COL_R2]))
                sim_feat_c = cosine01(ef1, ef2) if ef1 is not None and ef2 is not None else 0.0
                sim_rev_c = cosine01(er1, er2) if er1 is not None and er2 is not None else 0.0
                cscore = 0.5 * sim_feat_c + 0.5 * sim_rev_c

            if args.similarity_method == "jaccard":
                lscore = jscore
            elif args.similarity_method == "cosine":
                lscore = cscore
            else:
                beta = max(0.0, min(1.0, args.similarity_beta))
                lscore = beta * cscore + (1.0 - beta) * jscore

            lex.append(lscore)
            lex_j.append(jscore)
            lex_c.append(cscore)

            # Rule: copied/nearly identical reviews but low feature overlap tends to be false positive.
            copied_review_feature_divergence = 1.0 if (sim_rev_j > 0.95 and sim_feat_j < 0.50) else 0.0
            rule.append(copied_review_feature_divergence)

        e12 = np.array(e12)
        e21 = np.array(e21)
        c12 = np.array(c12)
        c21 = np.array(c21)
        lex = np.array(lex)
        lex_j = np.array(lex_j)
        lex_c = np.array(lex_c)
        rule = np.array(rule)

        for agg_name, agg_fn in aggs.items():
            print(f"  Aggregator: {agg_name}")
            if args.nli_score_mode == "contra_norm":
                d12 = normalize_entail_with_contradiction(e12, c12)
                d21 = normalize_entail_with_contradiction(e21, c21)
            else:
                d12 = e12
                d21 = e21
            nli_score = agg_fn(d12, d21)
            contradiction = np.maximum(c12, c21)
            for alpha in alphas:
                blended = alpha * nli_score + (1.0 - alpha) * lex
                for ct in contra_thresholds:
                    guarded = blended.copy()
                    guarded[contradiction >= ct] = 0.0
                    for penalty in rule_penalties:
                        final_score = guarded * (1.0 - penalty * rule)
                        tuned_th, cv_m = tune_threshold_cv(y, final_score, folds, args.objective, threshold_grid)
                        pred = (final_score >= tuned_th).astype(int)
                        full_m = metrics(y, pred)
                        obj = objective_value(full_m, args.objective)
                        config_rows.append(
                            {
                                "template": tmpl_name,
                                "aggregator": agg_name,
                                "alpha": alpha,
                                "contradiction_th": ct,
                                "rule_penalty": penalty,
                                "tuned_th": tuned_th,
                                "cv_acc": cv_m["acc"],
                                "cv_f1": cv_m["f1"],
                                "cv_kappa": cv_m["kappa"],
                                "cv_balanced_acc": cv_m["balanced_acc"],
                                "full_acc": full_m["acc"],
                                "full_f1": full_m["f1"],
                                "full_kappa": full_m["kappa"],
                                "full_balanced_acc": full_m["balanced_acc"],
                            }
                        )
                        if best_cfg is None or (obj, full_m["f1"], full_m["acc"]) > (
                            best_final_obj,
                            best_aux.get("best_f1", -1e9),
                            best_aux.get("best_acc", -1e9),
                        ):
                            best_final_obj = obj
                            best_cfg = Config(
                                template=tmpl_name,
                                aggregator=agg_name,
                                alpha=alpha,
                                contradiction_th=ct,
                                rule_penalty=penalty,
                                tuned_th=tuned_th,
                                cv_acc=cv_m["acc"],
                                cv_f1=cv_m["f1"],
                                cv_kappa=cv_m["kappa"],
                                cv_balanced_acc=cv_m["balanced_acc"],
                            )
                            best_score_vector = final_score.copy()
                            best_aux = {
                                "e12": e12.copy(),
                                "e21": e21.copy(),
                                "c12": c12.copy(),
                                "c21": c21.copy(),
                                "lex": lex.copy(),
                                "lex_jaccard": lex_j.copy(),
                                "lex_cosine": lex_c.copy(),
                                "rule": rule.copy(),
                                "nli_score": nli_score.copy(),
                                "contradiction": contradiction.copy(),
                                "best_f1": full_m["f1"],
                                "best_acc": full_m["acc"],
                            }

    cfg_df = pd.DataFrame(config_rows).sort_values(
        ["full_kappa", "full_f1", "full_acc", "cv_kappa"], ascending=False
    )
    cfg_df.to_csv(out_cfg, index=False)

    assert best_cfg is not None and best_score_vector is not None
    best_pred = (best_score_vector >= best_cfg.tuned_th).astype(int)
    best_m = metrics(y, best_pred)
    low_th, high_th = find_triage_thresholds(y, best_score_vector, args.min_pos_precision, args.min_neg_precision)

    row_df = eval_df.copy()
    if COL_FIAZ in row_df.columns:
        row_df[COL_FIAZ] = safe_int_series(row_df[COL_FIAZ])
    if COL_NAVEEN in row_df.columns:
        row_df[COL_NAVEEN] = safe_int_series(row_df[COL_NAVEEN])
    row_df["target_label"] = y
    row_df["e12"] = best_aux["e12"]
    row_df["e21"] = best_aux["e21"]
    row_df["c12"] = best_aux["c12"]
    row_df["c21"] = best_aux["c21"]
    row_df["nli_score"] = best_aux["nli_score"]
    row_df["lex_score"] = best_aux["lex"]
    row_df["lex_jaccard"] = best_aux["lex_jaccard"]
    row_df["lex_cosine"] = best_aux["lex_cosine"]
    row_df["contradiction"] = best_aux["contradiction"]
    row_df["rule_flag"] = best_aux["rule"]
    row_df["final_score"] = best_score_vector
    row_df["pred"] = best_pred
    row_df["pred"] = safe_int_series(row_df["pred"])
    row_df["target_label"] = safe_int_series(row_df["target_label"])
    row_df["rule_flag"] = safe_int_series(row_df["rule_flag"])

    triage = np.where(
        best_score_vector >= high_th,
        "auto_positive",
        np.where(best_score_vector <= low_th, "auto_negative", "needs_review"),
    )
    row_df["triage_label"] = triage

    row_df["llm_label"] = np.nan
    row_df["llm_confidence"] = np.nan
    row_df["llm_rationale"] = ""
    row_df["pred_final"] = row_df["pred"].copy()
    row_df["llm_override"] = 0

    if args.llm_judge:
        row_df = run_llm_judge_stage(row_df, args, best_cfg.tuned_th)

    row_df["pred_final"] = safe_int_series(row_df["pred_final"])
    row_df["llm_override"] = safe_int_series(row_df["llm_override"])
    row_df["match"] = (row_df["pred_final"] == row_df["target_label"]).astype(int)
    row_df.to_csv(out_rows, index=False)

    triage_df = row_df[
        [
            COL_F1,
            COL_R1,
            COL_F2,
            COL_R2,
            "target_label",
            "final_score",
            "pred",
            "pred_final",
            "llm_label",
            "llm_confidence",
            ]
    ].copy()
    triage_df["triage_label"] = row_df["triage_label"]
    triage_df.to_csv(out_triage, index=False)

    # Per-run ablation table (markdown + csv) 
    nli_only = best_aux["nli_score"]
    sim_only = best_aux["lex"]
    combined = best_cfg.alpha * nli_only + (1.0 - best_cfg.alpha) * sim_only
    guarded = combined.copy()
    guarded[np.array(best_aux["contradiction"]) >= best_cfg.contradiction_th] = 0.0
    final_sc = guarded * (1.0 - best_cfg.rule_penalty * np.array(best_aux["rule"]))

    variants = [
        ("NLI only", nli_only),
        ("Similarity only", sim_only),
        (f"Combined (alpha={best_cfg.alpha:.2f})", combined),
        (f"+ Contradiction guard (tau_c={best_cfg.contradiction_th:.2f})", guarded),
        (f"+ Rule penalty (lambda={best_cfg.rule_penalty:.2f}) [Final]", final_sc),
    ]

    ablation_rows = []
    for i, (name, sc) in enumerate(variants, start=1):
        th, m = best_threshold_and_metrics_by_kappa(np.asarray(sc), y)
        ablation_rows.append(
            {
                "#": i,
                "Model": name,
                "Best th": float(th),
                "Acc": m["acc"],
                "F1": m["f1"],
                "Kappa": m["kappa"],
                "BalAcc": m["balanced_acc"],
                "TP": int(m["tp"]),
                "TN": int(m["tn"]),
                "FP": int(m["fp"]),
                "FN": int(m["fn"]),
            }
        )

    # If LLM judge is enabled, add final post-LLM row (no threshold search here).
    if args.llm_judge:
        m_llm = metrics(y, row_df["pred_final"].to_numpy().astype(int))
        ablation_rows.append(
            {
                "#": len(ablation_rows) + 1,
                "Model": f"+ LLM judge ({args.llm_model}) [Final]",
                "Best th": np.nan,
                "Acc": m_llm["acc"],
                "F1": m_llm["f1"],
                "Kappa": m_llm["kappa"],
                "BalAcc": m_llm["balanced_acc"],
                "TP": int(m_llm["tp"]),
                "TN": int(m_llm["tn"]),
                "FP": int(m_llm["fp"]),
                "FN": int(m_llm["fn"]),
            }
        )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv = out_dir / "process_ablation_table.csv"
    ablation_md = out_dir / "process_ablation_table.md"
    ablation_df.to_csv(ablation_csv, index=False)

    md_lines = ablation_markdown_lines(ablation_df)
    ablation_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print_ablation_terminal(ablation_df)
    print(f"Saved ablation table csv: {ablation_csv}")
    print(f"Saved ablation table md: {ablation_md}")

    print("Best configuration")
    print(
        f"  template={best_cfg.template} aggregator={best_cfg.aggregator} alpha={best_cfg.alpha:.2f} "
        f"contradiction_th={best_cfg.contradiction_th:.2f} rule_penalty={best_cfg.rule_penalty:.2f} "
        f"threshold={best_cfg.tuned_th:.2f}"
    )
    print(
        f"  CV: acc={best_cfg.cv_acc:.4f} f1={best_cfg.cv_f1:.4f} "
        f"kappa={best_cfg.cv_kappa:.4f} bal_acc={best_cfg.cv_balanced_acc:.4f}"
    )
    print(
        f"  Full: acc={best_m['acc']:.4f} f1={best_m['f1']:.4f} "
        f"kappa={best_m['kappa']:.4f} bal_acc={best_m['balanced_acc']:.4f}"
    )
    print(f"Triage thresholds: low={low_th:.2f} high={high_th:.2f}")
    print(f"Saved config search: {out_cfg}")
    print(f"Saved row scores: {out_rows}")
    print(f"Saved triage labels: {out_triage}")


if __name__ == "__main__":
    main()
