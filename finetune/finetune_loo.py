"""
finetune_loo.py — Leave-One-Out fine-tuning for TIA Task C benchmark.

For each of 17 folds:
  1. Train Mistral-7B LoRA on the 16 other cases
  2. Save the adapter (adapter_model.safetensors + adapter_config.json)
  3. Upload to the pre-created Cloudflare Workers AI finetune slot
  4. Run inference on the held-out case using that finetune
  5. Save the prediction

Checkpointing: saves progress after each fold so the run can be resumed
if interrupted. Check loo_progress.json to see which folds are done.

Runtime: ~10-12 min/fold × 17 folds ≈ 3-3.5 hours on a 2g.20gb GPU.

Usage (run in JupyterHub terminal):
    cd ~/volume
    python3 finetune_loo.py

To resume after interruption just run again — completed folds are skipped.
"""

import gc, json, os, re, time
from pathlib import Path

import requests
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Config ─────────────────────────────────────────────────────────────────────
CF_ACCOUNT_ID = "e90bf5d749ab3d8283460cb8c66b64c6"
CF_API_TOKEN = "cfat_0i7D5qSmBigJWvJbIQXtHBZb0Ucfja8iIeCZLb3u62106ee0"
CF_BASE = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai"
CF_MODEL = "@cf/mistral/mistral-7b-instruct-v0.2-lora"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

H = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}

# Pre-created CF finetune IDs — one per LOO fold (held-out case_id → finetune_id)
FINETUNE_IDS = {
    "CEQ-CA-010282": "d7606f43-f62d-4549-9905-52ec340b4f4b",
    "CEQ-CA-051151": "800159d0-e7d2-41dd-ae96-1c0d9e3ae43f",
    "CEQ-CA-060428": "89450d5e-d9c2-4555-bbce-e74e9cd16119",
    "CEQ-CA-070065": "189c8d1f-8017-4853-8db6-cc1186e13777",
    "CEQ-CA-070271": "d582acd0-890a-4bfc-93bc-fff2722c5e05",
    "CEQ-CA-080060": "22da45ff-fa9c-438c-b215-00dd4029b9b3",
    "CEQ-CA-080668": "76e53ef3-4209-40e4-be8a-7484bedbc8a7",
    "CEQ-CA-101010": "706bde0b-11b9-46dd-90ee-be747e095be1",
    "CEQ-CA-120005": "a961720e-d361-42fc-a8d7-07bc4cfd9652",
    "MEPA-MA-16012": "e3ab4a30-292c-489a-80f0-265b80b35bab",
    "MEPA-MA-16024": "faee6066-8a2a-4fbc-94ba-3114423422d0",
    "MEPA-MA-16241": "ebb509f6-bb30-4407-93c6-9ddbe8e4564f",
    "MEPA-MA-16468": "43fe9ec6-23ec-44d4-940f-768eb5218285",
    "MEPA-MA-16504": "fd60b397-ffdc-4a71-80c3-f67da703b2b6",
    "MEPA-MA-16561": "4e94c812-4d3b-4d7c-8e39-855cf6afa7d2",
    "MEPA-MA-16608": "b638706c-39a3-4456-9e01-6cd3d563b8e3",
    "SFP-CA-p003": "b933f80b-385a-41d9-8cdf-ce1af2bb1a3c",
}

VALID_MEASURE_TYPES = {
    "add_turn_lane",
    "signal_timing_optimization",
    "new_traffic_signal",
    "access_modification",
    "driveway_consolidation",
    "road_widening",
    "roundabout_installation",
    "pedestrian_improvement",
    "transit_improvement",
    "tdm_measure",
    "no_mitigation_required",
    "other",
}

# ── Load training data ─────────────────────────────────────────────────────────
print("Loading LOO training data...")
all_pairs = json.load(open("finetune_loo_data.json"))
all_cases = [p["case_id"] for p in all_pairs]
print(f"  {len(all_pairs)} cases loaded: {all_cases}")

# ── Load checkpoint ────────────────────────────────────────────────────────────
PROGRESS_FILE = Path("loo_progress.json")
PREDS_FILE = Path("loo_predictions.json")

progress = json.loads(PROGRESS_FILE.read_text()) if PROGRESS_FILE.exists() else {}
predictions = json.loads(PREDS_FILE.read_text()) if PREDS_FILE.exists() else {}

completed = set(progress.keys())
remaining = [c for c in all_cases if c not in completed]
print(f"\nProgress: {len(completed)}/17 folds done. Remaining: {len(remaining)}")

# ── Load tokenizer and base model ONCE — reuse across all 17 folds ────────────
# Loading the 4-bit model takes ~2 min. Reloading each fold wastes time and
# causes GPU memory fragmentation (the OOM error). Instead, keep the base model
# loaded throughout and only add/remove the small LoRA adapter per fold.
tokenizer = None
base_model = None  # 4-bit base model — loaded once, never deleted


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_base_model():
    global base_model
    if base_model is None:
        print("\nLoading base model (4-bit) — this is done ONCE for all 17 folds...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model = prepare_model_for_kbit_training(base_model)
        print("  Base model loaded and ready.\n")
    return base_model


def format_mistral(messages):
    tok = get_tokenizer()
    user = next(m["content"] for m in messages if m["role"] == "user")
    asst = next(m["content"] for m in messages if m["role"] == "assistant")
    return f"[INST] {user} [/INST] {asst}{tok.eos_token}"


# ── Helper: upload adapter to CF ───────────────────────────────────────────────
def upload_adapter(finetune_id: str, adapter_dir: Path) -> bool:
    for fname in ["adapter_model.safetensors", "adapter_config.json"]:
        fpath = adapter_dir / fname
        if not fpath.exists():
            print(f"  MISSING {fname}")
            return False
        with open(fpath, "rb") as f:
            r = requests.post(
                f"{CF_BASE}/finetunes/{finetune_id}/finetune-assets",
                headers={"Authorization": f"Bearer {CF_API_TOKEN}"},
                files={"file": (fname, f)},
                data={"file_name": fname},
                timeout=300,
            )
        if r.status_code == 200:
            print(f"  ✓ {fname} uploaded")
        else:
            print(f"  ✗ {fname} failed: {r.status_code} {r.text[:100]}")
            return False
    return True


# ── Helper: run inference with fine-tuned model ────────────────────────────────
def run_finetuned_inference(finetune_id: str, user_msg: str) -> dict:
    r = requests.post(
        f"{CF_BASE}/run/{CF_MODEL}",
        headers={**H, "Content-Type": "application/json"},
        json={
            "messages": [
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "lora": finetune_id,
            "raw": True,
        },
        timeout=120,
    )
    if r.status_code != 200:
        print(f"  Inference failed: {r.status_code} {r.text[:200]}")
        return {"recommendations": []}

    data = r.json()
    result = data.get("result", {})
    text = result.get("response") or ""
    if isinstance(text, dict):
        parsed = text
    else:
        text = re.sub(r"^```(?:json)?\s*", "", str(text).strip())
        text = re.sub(r"\s*```$", "", text.strip())
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}

    # Normalise
    recs = parsed.get("recommendations", []) if isinstance(parsed, dict) else []
    out = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        mt = str(rec.get("measure_type", "other")).lower().replace(" ", "_")
        if mt not in VALID_MEASURE_TYPES:
            mt = "other"
        out.append(
            {
                "location": str(rec.get("location", "not_specified"))[:100],
                "measure_type": mt,
                "description": str(rec.get("description", ""))[:500],
                "timing": str(rec.get("timing", "not_specified")),
                "responsible_party": str(rec.get("responsible_party", "not_specified")),
            }
        )
    return {"recommendations": out}


# ── Main LOO loop ──────────────────────────────────────────────────────────────
# Pre-load the base model and tokenizer before entering the fold loop
get_tokenizer()
get_base_model()

for fold_idx, held_out_id in enumerate(all_cases):
    if held_out_id in completed:
        print(f"\n[{fold_idx + 1:2d}/17] SKIP {held_out_id} (already done)")
        continue

    print(f"\n{'=' * 70}")
    print(f"[{fold_idx + 1:2d}/17] Fold held-out: {held_out_id}")
    print(f"{'=' * 70}")

    finetune_id = FINETUNE_IDS[held_out_id]
    train_pairs = [p for p in all_pairs if p["case_id"] != held_out_id]
    test_pair = next(p for p in all_pairs if p["case_id"] == held_out_id)

    print(f"  Training on {len(train_pairs)} cases, testing on {held_out_id}")

    # ── Build tokenised dataset ────────────────────────────────────────────────
    tok = get_tokenizer()
    texts = [format_mistral(p["messages"]) for p in train_pairs]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        enc = tok(batch["text"], truncation=True, max_length=2048, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # ── Attach a fresh LoRA adapter to the pre-loaded base model ──────────────
    print("  Attaching LoRA adapter to base model...")
    base = get_base_model()
    base.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    # ── Train ──────────────────────────────────────────────────────────────────
    adapter_dir = Path(f"adapter_fold_{fold_idx:02d}")
    adapter_dir.mkdir(exist_ok=True)

    args = TrainingArguments(
        output_dir=str(adapter_dir / "checkpoints"),
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,  # fp16 conflicts with 4-bit quant + AMP scaler on A100
        bf16=True,  # A100 supports bf16 natively — use this instead
        logging_steps=5,
        save_strategy="no",
        optim="paged_adamw_8bit",
        warmup_steps=5,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
    )

    t0 = time.time()
    print(f"  Training...")
    trainer.train()
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s")

    # ── Save adapter ───────────────────────────────────────────────────────────
    model.save_pretrained(str(adapter_dir))

    # Convert to safetensors if needed
    bin_path = adapter_dir / "adapter_model.bin"
    st_path = adapter_dir / "adapter_model.safetensors"
    if bin_path.exists() and not st_path.exists():
        weights = torch.load(str(bin_path), map_location="cpu")
        save_file({k: v.contiguous() for k, v in weights.items()}, str(st_path))
        print("  Converted to safetensors")

    # Patch model_type in adapter_config.json
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["model_type"] = "mistral"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # ── Upload to CF ───────────────────────────────────────────────────────────
    print(f"  Uploading adapter to CF finetune {finetune_id[:8]}...")
    ok = upload_adapter(finetune_id, adapter_dir)
    if not ok:
        print(f"  ✗ Upload failed for fold {fold_idx} — skipping inference")
        continue

    # Brief pause for CF to process the upload
    time.sleep(5)

    # ── Inference on held-out case ─────────────────────────────────────────────
    print(f"  Running inference on {held_out_id}...")
    user_msg = next(m["content"] for m in test_pair["messages"] if m["role"] == "user")
    pred = run_finetuned_inference(finetune_id, user_msg)
    n_pred = len(pred.get("recommendations") or [])
    n_gt = len(test_pair["gt_recs"])
    print(f"  Prediction: {n_pred} recs  |  GT: {n_gt} recs")

    # ── Save progress ──────────────────────────────────────────────────────────
    predictions[held_out_id] = {
        "case_id": held_out_id,
        "fold_idx": fold_idx,
        "finetune_id": finetune_id,
        "prediction": pred,
        "gt_recs": test_pair["gt_recs"],
        "train_time_s": round(elapsed),
    }
    progress[held_out_id] = {"status": "done", "fold_idx": fold_idx}

    PREDS_FILE.write_text(json.dumps(predictions, indent=2))
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    print(f"  ✓ Fold {fold_idx + 1}/17 complete — progress saved")

    # Properly unload LoRA from base model so next fold starts clean
    # model.unload() removes the adapter layers and returns the unwrapped base
    del trainer
    base_model = model.unload()  # strips LoRA, returns clean base; update global
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(f"  LoRA unloaded. Base model ready for next fold.")

# ── Final summary ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"LOO COMPLETE: {len(predictions)}/17 folds")
print(f"{'=' * 70}")
print(f"\nPredictions saved to: loo_predictions.json")
print(f"Progress saved to:    loo_progress.json")
print()
print("Fold results:")
for case_id, p in predictions.items():
    n_pred = len(p["prediction"].get("recommendations") or [])
    n_gt = len(p["gt_recs"])
    print(
        f"  {case_id:25s}  pred={n_pred:2d}  gt={n_gt:2d}  train_time={p.get('train_time_s', '?')}s"
    )
