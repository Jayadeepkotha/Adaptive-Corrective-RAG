import re
import string
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

def normalize_answer(s: str) -> str:
    """Standard SQuAD/HotpotQA answer normalization: lowercase, strip punctuation & articles."""
    if not s:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction: str, gold_answers: List[str]) -> float:
    """Returns 1.0 if prediction exactly matches any gold answer after normalization, else 0.0."""
    norm_pred = normalize_answer(prediction)
    if not norm_pred:
        return 0.0
    for gold in gold_answers:
        if norm_pred == normalize_answer(gold):
            return 1.0
    return 0.0

def compute_token_f1(prediction: str, gold_answers: List[str]) -> float:
    """Computes max token-level F1 across all gold answer aliases using SQuAD logic."""
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            f1 = (2.0 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1

    return round(best_f1, 4)

def compute_contains_answer(prediction: str, gold_answers: List[str]) -> float:
    """Returns 1.0 if any normalized gold answer string is contained in the normalized prediction."""
    norm_pred = normalize_answer(prediction)
    if not norm_pred:
        return 0.0
    for gold in gold_answers:
        norm_gold = normalize_answer(gold)
        if norm_gold and norm_gold in norm_pred:
            return 1.0
    return 0.0

def compute_retrieval_recall(
    retrieved_sources: List[Dict[str, Any]],
    gold_supporting_titles: List[str]
) -> float:
    """
    Computes recall@k against the gold supporting paragraph titles.
    recall@k = (number of gold supporting titles found in retrieved chunks) / total_gold
    """
    if not gold_supporting_titles:
        return 1.0

    norm_gold_titles = [normalize_answer(t) for t in gold_supporting_titles if t]
    if not norm_gold_titles:
        return 1.0

    retrieved_titles = set()
    for s in retrieved_sources:
        doc_name = s.get("doc_name", "")
        # doc_name might be the title directly or title.txt
        clean_title = normalize_answer(doc_name.replace(".txt", "").replace(".pdf", ""))
        retrieved_titles.add(clean_title)

    hits = 0
    for gold_t in norm_gold_titles:
        # Check if gold_t matches any retrieved title or is substring
        if any(gold_t == rt or gold_t in rt or rt in gold_t for rt in retrieved_titles):
            hits += 1

    return round(1.0 * hits / len(norm_gold_titles), 4)

def classify_error(
    exact_match: float,
    contains_gold: float,
    retrieval_recall: float
) -> str:
    """
    Classifies question outcomes into mutually exclusive categories:
    - 'correct': Exact match achieved (EM == 1.0)
    - 'format_error': EM == 0.0 but contains_gold == 1.0 (answer is in prediction, but formatting/verbosity failed EM)
    - 'retrieval_miss': EM == 0.0, contains_gold == 0.0, and retrieval_recall < 1.0 (retriever failed to supply all gold context)
    - 'reasoning_error': EM == 0.0, contains_gold == 0.0, and retrieval_recall == 1.0 (all gold evidence present, but LLM failed to deduce correct answer)
    """
    if exact_match >= 1.0:
        return "correct"
    if contains_gold >= 1.0:
        return "format_error"
    if retrieval_recall < 1.0:
        return "retrieval_miss"
    return "reasoning_error"

def compute_bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Computes standard non-parametric percentile bootstrap confidence interval on percentage scale [0, 100].
    """
    if not scores:
        return 0.0, 0.0
    arr = np.array(scores, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    
    rng = np.random.RandomState(seed)
    n = len(arr)
    # Resample indices with replacement
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    bootstrap_means = arr[indices].mean(axis=1) * 100.0

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrap_means, 100.0 * alpha))
    upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha)))
    return round(lower, 2), round(upper, 2)

def compute_paired_bootstrap(
    scores_strat: List[float],
    scores_base: List[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired bootstrap difference and win/loss/tie counts between strategy and baseline.
    Differences are on percentage scale [0, 100].
    """
    if not scores_strat or not scores_base or len(scores_strat) != len(scores_base):
        return {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "diff_mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    arr_s = np.array(scores_strat, dtype=float)
    arr_b = np.array(scores_base, dtype=float)
    
    wins = int(np.sum(arr_s > arr_b))
    losses = int(np.sum(arr_s < arr_b))
    ties = int(np.sum(arr_s == arr_b))

    diffs = (arr_s - arr_b) * 100.0
    diff_mean = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    n = len(diffs)
    indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_diff_means = diffs[indices].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_diff_means, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_diff_means, 100.0 * (1.0 - alpha)))

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "diff_mean": round(diff_mean, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
    }


    return round(float(hits) / float(len(norm_gold_titles)), 4)
