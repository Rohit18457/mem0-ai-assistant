"""
Evaluation Suite for AI Memory Assistant.
Measures:
1. Retrieval Precision@k and Recall@k.
2. Retrieval Latency (ms).
3. Response Grounding (verifying LLM response incorporates retrieved memories).
"""

import time
from typing import List, Dict, Any
from memory_agent import MemoryAssistant

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False

TEST_USER = "benchmark_eval_user"

# Benchmark ground-truth facts
BENCHMARK_FACTS = [
    "User is an AI engineer with 4 years of experience in deep learning.",
    "User prefers building backend microservices in Go and Python.",
    "User lives in Pune, Maharashtra, India.",
    "User's favorite vector database is Qdrant.",
    "User loves drinking green tea while coding."
]

# Evaluation test cases: query, expected keywords in retrieved memory, expected answer keyword
EVAL_TESTS = [
    {
        "query": "What programming languages do I like to code in?",
        "expected_fact_keyword": "Python",
        "expected_answer_keyword": "python"
    },
    {
        "query": "Where do I currently reside?",
        "expected_fact_keyword": "Pune",
        "expected_answer_keyword": "pune"
    },
    {
        "query": "How many years have I worked in AI?",
        "expected_fact_keyword": "4 years",
        "expected_answer_keyword": "4"
    },
    {
        "query": "What is my favorite vector store?",
        "expected_fact_keyword": "Qdrant",
        "expected_answer_keyword": "qdrant"
    },
    {
        "query": "What beverage do I drink when programming?",
        "expected_fact_keyword": "green tea",
        "expected_answer_keyword": "tea"
    }
]


def run_evaluation():
    print(f"🚀 Initializing Evaluation Benchmark on user '{TEST_USER}'...\n")
    assistant = MemoryAssistant(user_id=TEST_USER)

    # 1. Clean slate
    assistant.clear_all_memories()

    # 2. Ingest benchmark ground-truth memories
    print("📥 Ingesting benchmark facts into Qdrant...")
    for fact in BENCHMARK_FACTS:
        assistant.mem_client.add(fact, user_id=TEST_USER, infer=False)
    time.sleep(1)  # Brief pause for indexing

    print(f"✓ Ingested {len(BENCHMARK_FACTS)} facts.\n")

    # 3. Evaluate retrieval and generation
    results_table = []
    latencies = []
    precision_hits = 0
    grounding_hits = 0
    k = 3

    for idx, test in enumerate(EVAL_TESTS, start=1):
        query = test["query"]
        expected_kw = test["expected_fact_keyword"].lower()
        expected_ans_kw = test["expected_answer_keyword"].lower()

        # Measure retrieval latency
        start_t = time.perf_counter()
        retrieved = assistant.search_memories(query, limit=k)
        latency_ms = (time.perf_counter() - start_t) * 1000
        latencies.append(latency_ms)

        # Check Precision@k: Did the retrieved list contain the expected ground truth?
        retrieved_combined = " ".join(retrieved).lower()
        retrieval_success = expected_kw in retrieved_combined
        if retrieval_success:
            precision_hits += 1

        # Check Generation Grounding: Does the LLM incorporate the fact?
        response, _, _ = assistant.chat(query)
        grounding_success = expected_ans_kw in response.lower()
        if grounding_success:
            grounding_hits += 1

        results_table.append({
            "test_num": idx,
            "query": query,
            "retrieved_top1": retrieved[0] if retrieved else "None",
            "retrieval_success": retrieval_success,
            "grounding_success": grounding_success,
            "latency_ms": latency_ms
        })

    # 4. Cleanup benchmark memories
    assistant.clear_all_memories()

    # 5. Compute Metrics
    total = len(EVAL_TESTS)
    accuracy_retrieval = (precision_hits / total) * 100
    accuracy_grounding = (grounding_hits / total) * 100
    avg_latency = sum(latencies) / len(latencies)

    # 6. Display Results
    if USE_RICH:
        table = Table(title="📊 Memory Retrieval & Grounding Evaluation", border_style="cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("Test Query", style="white")
        table.add_column("Top Retrieved Memory", style="green")
        table.add_column("Retrieval @k", justify="center")
        table.add_column("LLM Grounded", justify="center")
        table.add_column("Latency (ms)", justify="right")

        for row in results_table:
            r_icon = "[green]PASS[/green]" if row["retrieval_success"] else "[red]FAIL[/red]"
            g_icon = "[green]PASS[/green]" if row["grounding_success"] else "[red]FAIL[/red]"
            table.add_row(
                str(row["test_num"]),
                row["query"],
                row["retrieved_top1"][:50] + ("..." if len(row["retrieved_top1"]) > 50 else ""),
                r_icon,
                g_icon,
                f"{row['latency_ms']:.1f}"
            )
        console.print(table)

        summary = (
            f"[bold cyan]📈 Benchmark Summary:[/bold cyan]\n"
            f"• [bold white]Retrieval Recall @{k}:[/bold white] [green]{accuracy_retrieval:.1f}%[/green] ({precision_hits}/{total})\n"
            f"• [bold white]Response Grounding Rate:[/bold white] [green]{accuracy_grounding:.1f}%[/green] ({grounding_hits}/{total})\n"
            f"• [bold white]Average Retrieval Latency:[/bold white] [yellow]{avg_latency:.1f} ms[/yellow]"
        )
        console.print(Panel(summary, border_style="green", expand=False))
    else:
        print("\n--- Evaluation Results ---")
        for row in results_table:
            status = "PASS" if row["retrieval_success"] and row["grounding_success"] else "FAIL"
            print(f"[{status}] {row['query']} -> {row['latency_ms']:.1f}ms")
        print("\n--- Summary ---")
        print(f"Retrieval Recall@{k}: {accuracy_retrieval:.1f}%")
        print(f"Response Grounding: {accuracy_grounding:.1f}%")
        print(f"Average Latency: {avg_latency:.1f}ms")


if __name__ == "__main__":
    run_evaluation()
