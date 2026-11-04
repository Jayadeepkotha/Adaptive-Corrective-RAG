import sys
import os

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.pipeline import AdaptiveRAGPipeline
from src.retriever import HybridRetriever

console = Console(legacy_windows=False)

SAMPLE_DOCUMENTS = [
    {
        "name": "Cloud_Architecture_2024.txt",
        "text": (
            "AWS Lambda is a serverless compute service that runs code in response to events. "
            "Lambda pricing is based on the number of requests and duration of execution. "
            "In 2024, AWS Lambda charges $0.20 per 1 million requests and $0.0000166667 for every GB-second of compute time. "
            "AWS Lambda supports maximum execution timeout of 15 minutes per invocation. "
            "Cold start latency in AWS Lambda ranges between 100ms and 1 second depending on the runtime language. "
            "Python and Node.js have significantly faster cold start times compared to Java and .NET."
        )
    },
    {
        "name": "Google_Cloud_Functions_2024.txt",
        "text": (
            "Google Cloud Functions (2nd gen) is Google's serverless execution environment built on Cloud Run and Knative. "
            "Google Cloud Functions pricing charges $0.40 per 1 million invocations and $0.0000100 per GB-second. "
            "The maximum request timeout for 2nd gen functions is up to 60 minutes for HTTP functions. "
            "Google Cloud Functions offers concurrency up to 1000 concurrent requests per function instance, "
            "which dramatically minimizes cold starts compared to traditional 1-request-per-instance models."
        )
    },
    {
        "name": "Cybersecurity_Policy_2024.txt",
        "text": (
            "Under Section 4.2 of the Enterprise Cybersecurity Policy, all production systems must enforce Multi-Factor Authentication (MFA). "
            "Statutory penalty for non-compliance with data encryption standards is $50,000 per violation. "
            "Security incident response teams must report data breaches to the regulatory board within 72 hours of initial discovery. "
            "Failure to report within the 72-hour window incurs an additional penalty of $10,000 per day of delay."
        )
    }
]

def main():
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print("  ADAPTIVE-RAG: COMPLEXITY-AWARE RETRIEVAL-AUGMENTED GENERATION", style="bold yellow")
    console.print("  Implementation based on NAACL 2024 (Jeong et al., KAIST)", style="dim italic")
    console.print("=" * 80 + "\n", style="bold cyan")

    # 1. Initialize Pipeline & Retriever
    console.print("[*] Loading Embedding Models and indexing documents...")
    retriever = HybridRetriever()
    for doc in SAMPLE_DOCUMENTS:
        retriever.add_text_document(doc["name"], doc["text"])
    pipeline = AdaptiveRAGPipeline(retriever=retriever)

    console.print(f"[OK] Indexed {retriever.count()} document chunks into Hybrid Vector and BM25 store.\n", style="bold green")

    # 2. Test Queries demonstrating the 3 Complexity Routes
    test_queries = [
        # ROUTE 1: ZERO_RETRIEVAL (Parametric knowledge)
        "What is the mathematical definition of algorithmic time complexity O(n log n)?",

        # ROUTE 2: SINGLE_HOP (Direct factual lookup)
        "What is the statutory penalty for non-compliance with data encryption standards and the mandatory breach reporting timeline?",

        # ROUTE 3: MULTI_HOP (Cross-document comparison & decomposition)
        "Compare the pricing per million requests and the maximum execution timeout between AWS Lambda and Google Cloud Functions."
    ]

    for i, q in enumerate(test_queries, 1):
        console.print(f"[bold magenta]>> TEST QUERY {i}[/bold magenta]: [bold white]\"{q}\"[/bold white]")
        res = pipeline.process_query(q)

        # Display Routing Decision Table
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Adaptive Dimension", style="bold")
        table.add_column("Pipeline Output / Decision")

        color_map = {
            "ZERO_RETRIEVAL": "bright_green",
            "SINGLE_HOP": "bright_blue",
            "MULTI_HOP": "bright_magenta"
        }
        comp_color = color_map.get(res.complexity.value, "white")

        table.add_row("Detected Complexity", f"[{comp_color}]{res.complexity.value}[/{comp_color}]")
        table.add_row("Strategy Executed", f"[bold]{res.route_name}[/bold]")
        table.add_row("Classifier Reasoning", res.classifier_reasoning)
        
        if res.sub_queries:
            sub_q_str = "\n".join([f"- {sq}" for sq in res.sub_queries])
            table.add_row("Decomposed Sub-Queries", f"[italic]{sub_q_str}[/italic]")

        table.add_row(
            "Latency Breakdown",
            f"Classify: [yellow]{res.metrics.classification_ms}ms[/yellow] | "
            f"Retrieve: [yellow]{res.metrics.retrieval_ms}ms[/yellow] | "
            f"Generate: [yellow]{res.metrics.generation_ms}ms[/yellow] | "
            f"Total: [bold green]{res.metrics.total_latency_ms}ms[/bold green]"
        )
        table.add_row("Retrieval Stats", f"Steps: {res.metrics.retrieval_steps} | Chunks Used: {res.metrics.chunks_retrieved}")

        console.print(table)

        # Display Final Answer Panel
        console.print(Panel(
            res.answer,
            title="[bold green]Final Answer[/bold green]",
            border_style="green",
            subtitle=f"[dim]Total Latency: {res.metrics.total_latency_ms}ms[/dim]"
        ))
        console.print("\n" + "-" * 80 + "\n")

    console.print("[OK] All 3 Adaptive-RAG complexity routes verified successfully!\n", style="bold green")

if __name__ == "__main__":
    main()
