"""
run_benchmark.py — Orchestrator for the full TIA LLM benchmark pipeline.

Stages:
  0. embed    — precompute BGE-M3 chunk embeddings
  1. bench    — run Task A, A+RAG, B for all models
  2. evaluate — compute precision/recall/F1
  3. results  — produce summary.json and print leaderboard

Usage:
    python3 run_benchmark.py                      # full pipeline
    python3 run_benchmark.py --stages embed bench # specific stages
    python3 run_benchmark.py --models llama kimi  # specific models
    python3 run_benchmark.py --tasks A A_RAG      # specific tasks
    python3 run_benchmark.py --overwrite          # re-run existing predictions
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))
from config import LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "run_benchmark.log"),
    ],
)
log = logging.getLogger(__name__)

ALL_STAGES = ["embed", "bench", "evaluate", "results"]


def run_stage(name: str, **kwargs) -> None:
    t0 = time.time()
    log.info("▶  Stage: %s", name.upper())
    try:
        if name == "embed":
            from embed import run_embed

            # embed only the 33 evaluation cases → clean LOO corpus for Task C_RAG
            run_embed(
                batch_size=kwargs.get("batch_size", 20), resume=True, eval_only=True
            )

        elif name == "bench":
            from benchmark import run_benchmark

            run_benchmark(
                model_keys=kwargs.get("models"),
                tasks=kwargs.get("tasks"),
                overwrite=kwargs.get("overwrite", False),
            )

        elif name == "evaluate":
            from evaluate import run_evaluation

            run_evaluation(model_keys=kwargs.get("models"))

        elif name == "results":
            from results import run_results

            run_results()

        log.info("✓  Stage %s done in %.1fs\n", name.upper(), time.time() - t0)
    except SystemExit:
        raise
    except Exception as e:
        log.exception("✗  Stage %s FAILED: %s", name, e)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="TIA LLM Benchmark Pipeline")
    ap.add_argument(
        "--stages",
        nargs="+",
        default=ALL_STAGES,
        choices=ALL_STAGES,
        help="Stages to run (default: all)",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=[
            "llama-3.3-70b",
            "kimi-k2.5",
            "gpt-oss-120b",
            "nemotron-120b",
            "qwen3-30b",
            "gemma-3-12b",
        ],
        default=None,
        help="Models to benchmark (default: all)",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        choices=["A", "A_RAG", "B"],
        default=None,
        help="Tasks to run (default: all)",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Re-run predictions that already exist"
    )
    ap.add_argument(
        "--stages",
        nargs="+",
        default=ALL_STAGES,
        choices=ALL_STAGES,
        help="Stages to run (default: all)",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=[
            "llama-3.3-70b",
            "gpt-oss-120b",
            "nemotron-120b",
            "qwen3-30b",
            "gemma-3-12b",
        ],
        default=None,
        help="Models to benchmark (default: all)",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        choices=["main", "rag"],
        default=None,
        help="Tasks to run (default: C and C_RAG)",
    )
    args = ap.parse_args()

    log.info(
        "Pipeline: stages=%s  models=%s  tasks=%s",
        args.stages,
        args.models or "all",
        args.tasks or "all",
    )

    for stage in args.stages:
        run_stage(
            stage,
            models=args.models,
            tasks=args.tasks,
            overwrite=args.overwrite,
            batch_size=args.batch_size,
        )

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
