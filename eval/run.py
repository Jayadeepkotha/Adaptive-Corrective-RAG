import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from loguru import logger

from eval.cache import EvalCache
from eval.dataset import DatasetLoader, BenchmarkSample
from eval.metrics import (
    compute_exact_match,
    compute_token_f1,
    compute_contains_answer,
    compute_retrieval_recall,
    classify_error,
    compute_bootstrap_ci,
    compute_paired_bootstrap,
)
from src.pipeline import AdaptiveRAGPipeline

ALL_STRATEGIES = [
    "no_retrieval",
    "single_step",
    "multi_step",
    "adaptive",
    "adaptive_corrective",
    "oracle",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardized Benchmark Evaluation Harness for Adaptive + Corrective RAG"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "2wikimultihop", "musique"],
        help="Evaluation dataset to run (default: hotpotqa)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of questions to evaluate (default: 50)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Alias for --n to support quick smoke runs"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="tune",
        choices=["tune", "test", "all"],
        help="Dataset split: 'tune' (first 250 questions), 'test' (second 250 questions), or 'all' (500 questions) (default: tune)"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Comma-separated list of strategies, or 'all'. Options: "
             "no_retrieval, single_step, multi_step, adaptive, adaptive_corrective, oracle"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="distractor",
        choices=["distractor", "pooled"],
        help="Retrieval setting: 'distractor' (index only the question's candidate paragraphs) "
             "or 'pooled' (index all candidate paragraphs from all sampled questions together)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic question sampling (default: 42)"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Ablation: disable cross-encoder precision reranker"
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Ablation: disable Knowledge Graph traversal"
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Ablation: disable CRAG web search fallback"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable SQLite disk caching of LLM and retrieval calls"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results",
        help="Directory to save evaluation result JSON and CSV files"
    )
    return parser.parse_args()


def resolve_strategies(strat_arg: str) -> List[str]:
    strat_arg = strat_arg.strip().lower()
    if strat_arg == "all":
        return list(ALL_STRATEGIES)
    selected = [s.strip() for s in strat_arg.split(",") if s.strip()]
    invalid = [s for s in selected if s not in ALL_STRATEGIES]
    if invalid:
        raise ValueError(f"Invalid strategies: {invalid}. Supported: {ALL_STRATEGIES}")
    return selected


def print_comparison_table(results_summary: List[Dict[str, Any]], base_strat: str = "single_step"):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(
            title=f"Multi-Hop QA Benchmark Evaluation (Paired vs Baseline '{base_strat}')",
            box=box.ROUNDED,
            header_style="bold cyan"
        )
        table.add_column("Strategy", style="bold white")
        table.add_column("EM (%) [95% CI]", justify="right", style="green")
        table.add_column("F1 (%) [95% CI]", justify="right", style="green")
        table.add_column("Recall (%) [95% CI]", justify="right", style="magenta")
        table.add_column("W/L/T vs Base", justify="right", style="yellow")
        table.add_column("Avg Lat (s)", justify="right")
        table.add_column("LLM Calls", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Taxonomy (R-Miss / Reason / Format)", justify="right", style="cyan")
        table.add_column("Errors", justify="right", style="red")

        for s in results_summary:
            em_ci_str = f"{s['exact_match']:.1f}% [{s['em_ci'][0]:.1f}, {s['em_ci'][1]:.1f}]"
            f1_ci_str = f"{s['f1']:.1f}% [{s['f1_ci'][0]:.1f}, {s['f1_ci'][1]:.1f}]"
            rec_ci_str = f"{s['retrieval_recall']:.1f}% [{s['recall_ci'][0]:.1f}, {s['recall_ci'][1]:.1f}]"
            wl_str = s.get("paired_vs_baseline", {}).get("em_win_loss_tie", "-")
            tax = s.get("error_taxonomy", {})
            tax_str = f"{tax.get('retrieval_miss', 0)} / {tax.get('reasoning_error', 0)} / {tax.get('format_error', 0)}"

            table.add_row(
                s["strategy"],
                em_ci_str,
                f1_ci_str,
                rec_ci_str,
                wl_str,
                f"{s['avg_latency_s']:.2f}s",
                f"{s['avg_llm_calls']:.1f}",
                f"{s['avg_total_tokens']:.0f}",
                tax_str,
                str(s["failures"])
            )
        console.print()
        console.print(table)
        console.print()
    except Exception:
        headers = ["Strategy", "EM (95% CI)", "F1 (95% CI)", "Recall (95% CI)", "W/L/T", "Avg Lat (s)", "Calls", "Tokens", "Taxonomy", "Err"]
        row_fmt = "{:<20} {:>18} {:>18} {:>20} {:>10} {:>10} {:>6} {:>8} {:>18} {:>4}"
        print("\n" + "=" * 145)
        print(row_fmt.format(*headers))
        print("-" * 145)
        for s in results_summary:
            em_ci_str = f"{s['exact_match']:.1f}% [{s['em_ci'][0]:.1f},{s['em_ci'][1]:.1f}]"
            f1_ci_str = f"{s['f1']:.1f}% [{s['f1_ci'][0]:.1f},{s['f1_ci'][1]:.1f}]"
            rec_ci_str = f"{s['retrieval_recall']:.1f}% [{s['recall_ci'][0]:.1f},{s['recall_ci'][1]:.1f}]"
            wl_str = s.get("paired_vs_baseline", {}).get("em_win_loss_tie", "-")
            tax = s.get("error_taxonomy", {})
            tax_str = f"{tax.get('retrieval_miss', 0)}/{tax.get('reasoning_error', 0)}/{tax.get('format_error', 0)}"
            print(row_fmt.format(
                s["strategy"],
                em_ci_str,
                f1_ci_str,
                rec_ci_str,
                wl_str,
                f"{s['avg_latency_s']:.2f}s",
                f"{s['avg_llm_calls']:.1f}",
                f"{s['avg_total_tokens']:.0f}",
                tax_str,
                str(s["failures"])
            ))
        print("=" * 145 + "\n")


def run_evaluation(args) -> Dict[str, Any]:
    n_samples = args.limit if args.limit is not None else args.n
    strategies = resolve_strategies(args.strategies)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = None
    if not args.no_cache:
        cache = EvalCache()

    logger.info(
        f"Starting Benchmark Evaluation: dataset={args.dataset}, split={args.split}, n={n_samples}, "
        f"setting={args.setting}, seed={args.seed}, strategies={strategies}"
    )

    loader = DatasetLoader()
    samples: List[BenchmarkSample] = loader.load_samples(
        args.dataset,
        n=n_samples,
        seed=args.seed,
        split=args.split
    )

    if args.no_graph:
        from src.config import settings
        settings.ENABLE_GRAPH_RAG = False

    pipeline = AdaptiveRAGPipeline(
        enable_graph=not args.no_graph,
        enable_rerank=not args.no_rerank,
        enable_web=not args.no_web
    )
    if cache is not None:
        pipeline.classifier.cache = cache
        pipeline.generator.cache = cache

    # In pooled setting: index all candidate passages from all sampled questions once
    if args.setting == "pooled":
        logger.info(f"Indexing pooled corpus across all {len(samples)} samples...")
        pipeline.retriever.clear()
        seen_keys = set()
        pooled_passages = []
        for s in samples:
            for ctx in s.contexts:
                t = ctx["title"]
                txt = ctx["text"]
                key = (t, txt)
                if key not in seen_keys:
                    seen_keys.add(key)
                    pooled_passages.append((t, txt))
        indexed_count = pipeline.retriever.add_passages(pooled_passages)
        logger.info(f"Pooled index built with {indexed_count} chunks.")

    strategy_summaries = []
    detailed_results = {}
    raw_metric_vectors = {}

    for strat in strategies:
        logger.info(f"\n=======================================================")
        logger.info(f"Evaluating Strategy: '{strat}' on {len(samples)} samples (split='{args.split}')...")
        logger.info(f"=======================================================")

        em_list = []
        f1_list = []
        contains_list = []
        recall_list = []
        latencies_s = []
        cached_list = []
        llm_calls_list = []
        tokens_list = []
        router_counts = {"ZERO_RETRIEVAL": 0, "SINGLE_HOP": 0, "MULTI_HOP": 0}
        crag_counts = {"CORRECT": 0, "AMBIGUOUS": 0, "INCORRECT": 0, "NONE": 0}
        taxonomy_counts = {"correct": 0, "format_error": 0, "retrieval_miss": 0, "reasoning_error": 0}
        web_search_count = 0
        failures = 0

        strat_per_question = []

        for idx, sample in enumerate(samples, 1):
            # In distractor setting: index ONLY candidate passages for this question
            # For oracle strategy: index strictly the gold supporting passages!
            if args.setting == "distractor":
                pipeline.retriever.clear()
                if strat == "oracle":
                    passages = [(c["title"], c["text"]) for c in sample.contexts if c.get("is_supporting", False)]
                else:
                    passages = [(c["title"], c["text"]) for c in sample.contexts]
                pipeline.retriever.add_passages(passages)

            pipeline.reset_counters()
            t0 = time.perf_counter()

            err_msg = None
            try:
                response = pipeline.process_query(
                    sample.question,
                    strategy=strat,
                    eval_mode=True
                )
                pred_answer = response.answer
                retrieved_sources = response.retrieved_sources
                complexity_name = response.complexity.name if hasattr(response.complexity, "name") else str(response.complexity)
                
                # Robust CRAG verdict normalization
                raw_verdict = response.crag_verdict
                if raw_verdict:
                    if isinstance(raw_verdict, str):
                        v = raw_verdict.strip().upper()
                        if "CORRECT" in v and "INCORRECT" not in v:
                            crag_verdict_name = "CORRECT"
                        elif "AMBIGUOUS" in v:
                            crag_verdict_name = "AMBIGUOUS"
                        elif "INCORRECT" in v:
                            crag_verdict_name = "INCORRECT"
                        else:
                            crag_verdict_name = v
                    elif hasattr(raw_verdict, "name"):
                        crag_verdict_name = raw_verdict.name
                    elif hasattr(raw_verdict, "value"):
                        crag_verdict_name = raw_verdict.value
                    else:
                        crag_verdict_name = "NONE"
                else:
                    crag_verdict_name = "NONE"

                web_triggered = len(response.web_sources) > 0 or (response.metrics and response.metrics.web_searches_triggered > 0)
            except Exception as e:
                logger.error(f"Error processing Q{idx} ('{sample.id}'): {e}")
                failures += 1
                err_msg = str(e)
                pred_answer = ""
                retrieved_sources = []
                complexity_name = "ERROR"
                crag_verdict_name = "ERROR"
                web_triggered = False

            latency_s = time.perf_counter() - t0
            latencies_s.append(latency_s)

            llm_stats = pipeline.get_llm_stats()
            is_cached = bool(llm_stats.get("is_cached", False))
            cached_list.append(is_cached)
            llm_calls_list.append(llm_stats["llm_calls"])
            tokens_list.append(llm_stats["total_tokens"])

            # Pacing delay between uncached benchmark queries to avoid cloud API rate limit bursts
            if not is_cached and idx < len(samples):
                time.sleep(1.0)

            # Compute QA Metrics
            em = compute_exact_match(pred_answer, sample.gold_answers)
            f1 = compute_token_f1(pred_answer, sample.gold_answers)
            contains = compute_contains_answer(pred_answer, sample.gold_answers)

            # Compute Retrieval Recall (only meaningful if retrieval occurred)
            if strat == "no_retrieval":
                recall = 0.0
            elif strat == "oracle":
                recall = 1.0
            else:
                recall = compute_retrieval_recall(retrieved_sources, sample.gold_supporting_titles)

            em_list.append(em)
            f1_list.append(f1)
            contains_list.append(contains)
            recall_list.append(recall)

            error_cat = classify_error(em, contains, recall)
            taxonomy_counts[error_cat] += 1

            if complexity_name in router_counts:
                router_counts[complexity_name] += 1
            if crag_verdict_name in crag_counts:
                crag_counts[crag_verdict_name] += 1
            if web_triggered:
                web_search_count += 1

            strat_per_question.append({
                "sample_id": sample.id,
                "question": sample.question,
                "gold_answers": sample.gold_answers,
                "gold_titles": sample.gold_supporting_titles,
                "prediction": pred_answer,
                "exact_match": em,
                "f1": f1,
                "contains_gold": contains,
                "retrieval_recall": recall,
                "error_category": error_cat,
                "latency_s": round(latency_s, 3),
                "is_cached": is_cached,
                "llm_stats": llm_stats,
                "router_complexity": complexity_name,
                "crag_verdict": crag_verdict_name,
                "web_search_triggered": web_triggered,
                "error": err_msg
            })

            if idx % 10 == 0 or idx == len(samples):
                logger.info(
                    f"[{strat}] Progress: {idx}/{len(samples)} | "
                    f"Cur Avg EM: {np.mean(em_list)*100:.1f}% | "
                    f"F1: {np.mean(f1_list)*100:.1f}% | "
                    f"Avg Lat: {np.mean(latencies_s):.2f}s"
                )

        # Latency accounting: report latency from uncached calls whenever available
        uncached_latencies = [lat for lat, cached in zip(latencies_s, cached_list) if not cached]
        if uncached_latencies:
            avg_lat = round(float(np.mean(uncached_latencies)), 3)
            p95_lat = round(float(np.percentile(uncached_latencies, 95)), 3)
        else:
            avg_lat = round(float(np.mean(latencies_s)), 3) if latencies_s else 0.0
            p95_lat = round(float(np.percentile(latencies_s, 95)), 3) if latencies_s else 0.0

        # Calculate Bootstrap 95% Confidence Intervals
        em_ci = compute_bootstrap_ci(em_list, seed=args.seed)
        f1_ci = compute_bootstrap_ci(f1_list, seed=args.seed)
        rec_ci = compute_bootstrap_ci(recall_list, seed=args.seed)

        n_total = len(samples)
        raw_metric_vectors[strat] = {
            "em": em_list,
            "f1": f1_list,
            "recall": recall_list
        }

        summary = {
            "strategy": strat,
            "dataset": args.dataset,
            "split": args.split,
            "setting": args.setting,
            "n_samples": n_total,
            "exact_match": round(float(np.mean(em_list) * 100), 2) if em_list else 0.0,
            "em_ci": list(em_ci),
            "f1": round(float(np.mean(f1_list) * 100), 2) if f1_list else 0.0,
            "f1_ci": list(f1_ci),
            "contains_gold": round(float(np.mean(contains_list) * 100), 2) if contains_list else 0.0,
            "retrieval_recall": round(float(np.mean(recall_list) * 100), 2) if recall_list else 0.0,
            "recall_ci": list(rec_ci),
            "avg_latency_s": avg_lat,
            "p95_latency_s": p95_lat,
            "uncached_samples_count": len(uncached_latencies),
            "cached_samples_count": n_total - len(uncached_latencies),
            "avg_llm_calls": round(float(np.mean(llm_calls_list)), 2) if llm_calls_list else 0.0,
            "avg_total_tokens": round(float(np.mean(tokens_list)), 1) if tokens_list else 0.0,
            "failures": failures,
            "error_taxonomy": taxonomy_counts,
            "router_distribution": router_counts,
            "router_accuracy_multihop_ref": round(float(router_counts.get("MULTI_HOP", 0) / max(1, n_total) * 100), 2),
            "crag_distribution": crag_counts,
            "web_searches_triggered": web_search_count
        }

        strategy_summaries.append(summary)
        detailed_results[strat] = {
            "summary": summary,
            "samples": strat_per_question
        }

    # Reference baseline selection for paired win/loss bootstrap
    base_strat = "single_step" if "single_step" in raw_metric_vectors else strategies[0]
    for summary in strategy_summaries:
        strat = summary["strategy"]
        if strat in raw_metric_vectors and base_strat in raw_metric_vectors:
            paired_em = compute_paired_bootstrap(
                raw_metric_vectors[strat]["em"],
                raw_metric_vectors[base_strat]["em"],
                seed=args.seed
            )
            paired_f1 = compute_paired_bootstrap(
                raw_metric_vectors[strat]["f1"],
                raw_metric_vectors[base_strat]["f1"],
                seed=args.seed
            )
            summary["paired_vs_baseline"] = {
                "baseline_strategy": base_strat,
                "em_diff_mean": paired_em["diff_mean"],
                "em_diff_ci": [paired_em["ci_lower"], paired_em["ci_upper"]],
                "em_win_loss_tie": f"{paired_em['wins']}/{paired_em['losses']}/{paired_em['ties']}",
                "f1_diff_mean": paired_f1["diff_mean"],
                "f1_diff_ci": [paired_f1["ci_lower"], paired_f1["ci_upper"]],
                "f1_win_loss_tie": f"{paired_f1['wins']}/{paired_f1['losses']}/{paired_f1['ties']}"
            }

    # Print comparison table
    print_comparison_table(strategy_summaries, base_strat=base_strat)

    # Save detailed JSON results
    json_path = out_dir / f"{timestamp}_{args.dataset}_{args.split}_{args.setting}.json"
    full_payload = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": args.dataset,
            "split": args.split,
            "setting": args.setting,
            "n_samples": n_samples,
            "seed": args.seed,
            "sample_ids": [s.id for s in samples],
            "baseline_strategy": base_strat,
            "ablations": {
                "no_rerank": args.no_rerank,
                "no_graph": args.no_graph,
                "no_web": args.no_web
            }
        },
        "summaries": strategy_summaries,
        "details": detailed_results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed evaluation results saved to: {json_path}")

    # Save summary CSV
    csv_path = out_dir / f"{timestamp}_{args.dataset}_{args.split}_{args.setting}.csv"
    csv_fields = [
        "strategy", "dataset", "split", "setting", "n_samples", "exact_match", "em_ci_lower", "em_ci_upper",
        "f1", "f1_ci_lower", "f1_ci_upper", "contains_gold", "retrieval_recall", "recall_ci_lower", "recall_ci_upper",
        "em_win_loss_tie", "avg_latency_s", "p95_latency_s", "avg_llm_calls", "avg_total_tokens", "failures",
        "taxonomy_correct", "taxonomy_format_error", "taxonomy_retrieval_miss", "taxonomy_reasoning_error",
        "router_accuracy_multihop_ref", "web_searches_triggered"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for s in strategy_summaries:
            row = dict(s)
            row["em_ci_lower"] = s.get("em_ci", [0, 0])[0]
            row["em_ci_upper"] = s.get("em_ci", [0, 0])[1]
            row["f1_ci_lower"] = s.get("f1_ci", [0, 0])[0]
            row["f1_ci_upper"] = s.get("f1_ci", [0, 0])[1]
            row["recall_ci_lower"] = s.get("recall_ci", [0, 0])[0]
            row["recall_ci_upper"] = s.get("recall_ci", [0, 0])[1]
            row["em_win_loss_tie"] = s.get("paired_vs_baseline", {}).get("em_win_loss_tie", "-")
            tax = s.get("error_taxonomy", {})
            row["taxonomy_correct"] = tax.get("correct", 0)
            row["taxonomy_format_error"] = tax.get("format_error", 0)
            row["taxonomy_retrieval_miss"] = tax.get("retrieval_miss", 0)
            row["taxonomy_reasoning_error"] = tax.get("reasoning_error", 0)
            writer.writerow(row)
    logger.info(f"Summary CSV saved to: {csv_path}")

    return full_payload


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
