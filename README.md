# TIA-LLM-Benchmark

**Can Large Language Models Generate Transportation Impact Analysis Recommendations?**
A benchmark study on conditioned mitigation generation using publicly available environmental review documents.

> *Paper submitted to Artificial Intelligence for Transportation*

---

## Overview

This repository contains:
- The **TIA benchmark dataset** (33 validated records from California and Massachusetts)
- The **dataset construction pipeline** (CEQAnet + MEPA programmatic scraping)
- The **benchmark evaluation framework** (zero-shot, few-shot, RAG, LoRA fine-tuning)
- The **paper source** (LaTeX)

### Key finding
Few-shot prompting with 2–3 complete demonstrations (F1_loc 0.118–0.124) outperforms zero-shot (0.070–0.109), RAG (0.048–0.141), and LOO LoRA fine-tuning (0.050) at this dataset scale — establishing that in-context learning is more sample-efficient than parameter updates when fewer than 17 in-domain examples are available.

---

## Repository Structure

```
tia-llm-benchmark/
├── data/
│   ├── final/tia_dataset.jsonl        ← 33-record benchmark dataset
│   ├── finetune_loo_data.json          ← 17 Task C training pairs (LOO)
│   └── finetune_loo_predictions.json  ← LOO fine-tuning predictions
│
├── pipeline/                           ← Dataset construction
│   ├── config.py                       ← Paths, patterns, config
│   ├── ceqanet_scraper.py              ← California CEQA scraper
│   ├── mepa_scraper.py                 ← Massachusetts MEPA API client
│   ├── sfplanning_crawler.py           ← SF Planning CDN crawler
│   ├── tia_hunter.py                   ← MuniCode meeting packet scanner
│   ├── parse.py                        ← PDF section extraction
│   ├── chunker.py                      ← Section-aware chunking
│   ├── embed.py                        ← BGE-M3 embedding generation
│   ├── extract.py                      ← CF Workers AI structured extraction
│   ├── build_dataset.py                ← Final JSONL assembly
│   ├── qa.py                           ← Quality assurance checks
│   └── summary.py                      ← Dataset statistics
│
├── benchmark/                          ← Evaluation pipeline
│   ├── benchmark.py                    ← Zero-shot / few-shot / RAG inference
│   ├── evaluate.py                     ← Precision / recall / F1 metrics
│   ├── results.py                      ← Leaderboard and summary
│   ├── baselines.py                    ← Frequency baseline + LLM-as-judge
│   └── run_benchmark.py                ← Orchestrator
│
├── finetune/
│   ├── finetune_loo.py                 ← LOO LoRA fine-tuning script (GPU)
│   └── finetune_notebook.ipynb         ← Single-fold Jupyter notebook
│
├── paper/
│   ├── main.tex                        ← LaTeX source
│   └── references.bib                  ← Bibliography
│
└── requirements.txt
```

---

## Dataset

`data/final/tia_dataset.jsonl` — 33 records, one JSON object per line.

Each record has the structure:
```json
{
  "case_id": "CEQ-CA-010282",
  "input": {
    "project_type": "mixed_use",
    "location_context": { "city": "Santa Clara", "state": "CA", ... },
    "description": { "narrative": "...", "trip_generation_summary": {...} },
    "known_conditions": { "study_intersections": [...], ... }
  },
  "ground_truth": {
    "issues": [
      { "location": "Lafayette St and El Camino Real", "los": "D",
        "vc_ratio": 0.88, "scenario": "pm_peak_build", ... }
    ],
    "recommendations": [
      { "location": "Future developments", "measure_type": "access_modification",
        "description": "...", "timing": "prior_to_occupancy", ... }
    ]
  },
  "metadata": {
    "source_url": "https://ceqanet.opr.ca.gov/...",
    "agency": "...", "year": 2022, "state": "CA",
    "exclude_from_retrieval": ["CEQ-CA-010282"]
  }
}
```

**17 of 33 records** have both non-empty `issues` and `recommendations` — these are the Task C evaluation cases.

**Sources:**
- 23 records from California CEQAnet (2018–2024)
- 10 records from Massachusetts MEPA (2021–2023)

---

## Setup

```bash
pip install -r requirements.txt
```

Create `pipeline/.env` with your Cloudflare credentials:
```
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_api_token
CF_MODEL=@cf/meta/llama-3.3-70b-instruct-fp8-fast
```

---

## Reproducing the Benchmark

### Step 1 — Generate embeddings (for RAG)
```bash
cd pipeline
python3 embed.py --eval-only --batch-size 20
```

### Step 2 — Run all benchmark conditions
```bash
cd benchmark
python3 benchmark.py --tasks main fewshot rag
```

### Step 3 — Evaluate and produce results
```bash
python3 run_benchmark.py --stages evaluate results
```

Results are saved to `data/results/`.

---

## Reproducing the Dataset

The dataset was constructed from two public sources. To reproduce from scratch:

### California CEQAnet
```bash
cd pipeline
python3 ceqanet_scraper.py --max-downloads 40 --year-start 2018 --year-end 2023
```

### Massachusetts MEPA
```bash
python3 mepa_scraper.py --years 2020 2021 2022 2023 --max-downloads 40
```

### Parse, extract, and build
```bash
python3 run_benchmark.py --stages parse chunk split build qa summary
```

> Note: The MEPA scraper requires the portal to be visited first to establish a
> server-side session (`mepa_scraper.py` handles this automatically via `warm_session()`).

---

## LoRA Fine-tuning

The LOO fine-tuning requires a GPU with ≥16 GB VRAM (tested on NVIDIA A100 MIG 2g.20gb).

```bash
# Install additional dependencies
pip install transformers peft bitsandbytes datasets accelerate safetensors

# Run LOO fine-tuning (17 folds, ~2.5 hours on A100)
cd finetune
python3 finetune_loo.py
```

The script checkpoints after each fold to `loo_progress.json` and can be resumed if interrupted.

---

## Evaluation Metrics

**F1_loc (primary):** Two-tier matching requiring both location agreement (Jaccard ≥ 0.5 on intersection name tokens) and exact measure_type match. Falls back to type-only matching for project-wide recommendations (location = "not_specified").

**F1_type (secondary):** Multi-set intersection on measure_type only, regardless of location.

**LLM-as-judge:** Llama-3.3-70B scores predictions on domain plausibility, location specificity, and GT alignment (1–5 scale).

---

## Results Summary

| Condition | Best F1_loc | Best model |
|---|---|---|
| Zero-shot | 0.109 | Qwen3-30B |
| Few-shot | 0.124 | Llama-3.3-70B |
| RAG | 0.141 | Gemma-3-12B |
| LoRA fine-tuned (LOO) | 0.050 | Mistral-7B |
| Frequency baseline | 0.249* | — |

*Metric artefact: baseline never names locations, exploits project-wide GT via Tier-2 fallback. Judge score: 1.88/5 vs 3.35–3.64/5 for LLMs.

---

## Citation

```bibtex
@article{bae2026tia,
  title   = {Can Large Language Models Generate Transportation Impact Analysis
             Recommendations? A Benchmark Study on Conditioned Mitigation
             Generation Using Publicly Available Environmental Review Documents},
  author  = {Bae, Andrew},
  journal = {Artificial Intelligence for Transportation},
  year    = {2026}
}
```

---

## License

Dataset construction code: MIT License.

Source documents are publicly available from:
- California Governor's Office of Planning and Research: [ceqanet.opr.ca.gov](https://ceqanet.opr.ca.gov)
- Massachusetts Executive Office of Energy and Environmental Affairs: [eeaonline.eea.state.ma.us](https://eeaonline.eea.state.ma.us)
