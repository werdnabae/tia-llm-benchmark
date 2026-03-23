"""
embed.py — Precompute BGE-M3 embeddings for every chunk in data/chunks/.

Model: @cf/baai/bge-m3 (1024-dimensional dense vectors)

Output: data/embeddings/chunks.json
  List of objects:
    { case_id, chunk_index, section_name, role, text, embedding: [float × 1024] }

Usage:
    python3 embed.py
    python3 embed.py --batch-size 50 --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

# Load .env
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from config import CHUNKS_DIR, LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "embed.log")],
)
log = logging.getLogger(__name__)

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
EMBED_MODEL = "@cf/baai/bge-m3"
EMBED_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{EMBED_MODEL}"

EMBED_DIR = Path(__file__).parent.parent / "data" / "embeddings"
OUT_PATH = EMBED_DIR / "chunks.json"  # original (all 86 parsed files)
OUT_PATH_EVAL = EMBED_DIR / "chunks_eval.json"  # clean LOO corpus (33 eval cases only)
DATASET_PATH = Path(__file__).parent.parent / "data" / "final" / "tia_dataset.jsonl"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
)


def embed_batch(texts, retries: int = 3):
    """
    Embed a batch of texts using BGE-M3.
    Returns list of 1024-dim vectors.
    """
    for attempt in range(1, retries + 2):
        try:
            resp = SESSION.post(
                EMBED_URL,
                # BGE-M3 accepts {"text": str} or {"text": [str, ...]}
                json={"text": texts[0] if len(texts) == 1 else texts},
                timeout=60,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                log.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            result = data.get("result", {})

            # BGE-M3 single: {"result": {"data": [[...]]}} or {"result": {"data": [...]}}
            # BGE-M3 batch:  {"result": {"data": [[...], [...], ...]}}
            raw = result.get("data") or result.get("embeddings") or result
            if isinstance(raw, list):
                if raw and isinstance(raw[0], list):
                    return raw  # list of vectors
                elif raw and isinstance(raw[0], float):
                    return [raw]  # single vector wrapped
            raise RuntimeError(f"Unexpected embedding response shape: {str(raw)[:200]}")

        except (requests.Timeout, requests.ConnectionError) as e:
            log.warning("Embed attempt %d failed: %s", attempt, e)
            if attempt > retries:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("All embed retries exhausted")


def load_all_chunks(eval_only: bool = False) -> list[dict]:
    """
    Load chunks from data/chunks/*.json.
    If eval_only=True, restrict to only the 33 cases in tia_dataset.jsonl
    (clean LOO corpus for Task C_RAG).
    """
    eval_ids: set = set()
    if eval_only:
        records = [
            json.loads(l) for l in DATASET_PATH.read_text().splitlines() if l.strip()
        ]
        eval_ids = {r["case_id"] for r in records}
        log.info("eval_only mode: restricting to %d evaluation case IDs", len(eval_ids))

    all_chunks = []
    for f in sorted(CHUNKS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        case_id = data.get("case_id") or f.stem
        if eval_only and case_id not in eval_ids:
            continue
        for chunk in data.get("chunks", []):
            all_chunks.append(
                {
                    "case_id": chunk["case_id"],
                    "chunk_index": chunk["chunk_index"],
                    "section_name": chunk["section_name"],
                    "subsection_name": chunk.get("subsection_name"),
                    "role": chunk["role"],
                    "text": chunk["text"],
                }
            )

    source = "evaluation cases only" if eval_only else "all files"
    log.info("Loaded %d chunks (%s)", len(all_chunks), source)
    return all_chunks


def run_embed(
    batch_size: int = 50, resume: bool = True, eval_only: bool = False
) -> None:
    out = OUT_PATH_EVAL if eval_only else OUT_PATH
    chunks = load_all_chunks(eval_only=eval_only)

    # Load already-embedded chunks if resuming
    embedded: dict = {}
    if resume and out.exists():
        existing = json.loads(out.read_text())
        for item in existing:
            key = (item["case_id"], item["chunk_index"])
            embedded[key] = item["embedding"]
        log.info("Resuming: %d chunks already embedded", len(embedded))

    # Filter to chunks that still need embedding
    pending = [c for c in chunks if (c["case_id"], c["chunk_index"]) not in embedded]
    log.info("Chunks to embed: %d", len(pending))

    total = len(pending)
    processed = 0

    for i in range(0, total, batch_size):
        batch = pending[i : i + batch_size]
        texts = [
            c["text"][:800] for c in batch
        ]  # ~200 tokens each; stay under 60K batch limit

        log.info("Embedding batch %d–%d / %d", i + 1, min(i + batch_size, total), total)
        vectors = embed_batch(texts)

        for chunk, vec in zip(batch, vectors):
            embedded[(chunk["case_id"], chunk["chunk_index"])] = vec

        # Save checkpoint every batch
        _save(chunks, embedded, out)
        processed += len(batch)
        time.sleep(0.2)  # polite delay

    _save(chunks, embedded, out)
    log.info("Done. %d / %d chunks embedded → %s", len(embedded), len(chunks), out)


def _save(chunks: list[dict], embedded: dict, out_path: Path) -> None:
    output = []
    for c in chunks:
        key = (c["case_id"], c["chunk_index"])
        if key in embedded:
            output.append({**c, "embedding": embedded[key]})
    out_path.write_text(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="Only embed the 33 evaluation cases (clean LOO corpus)",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    run_embed(batch_size=args.batch_size, resume=args.resume, eval_only=args.eval_only)
