#!/usr/bin/env python3
"""
tune.py

End-to-end LoRA fine-tuning on Apple Silicon using MLX-LM.

Inputs:
  - Clean CSV with columns: split,report,presence,severity,change
  - A prompt/system file (used as the chat system message)
  - Base model id/path compatible with MLX-LM (e.g. mlx-community/medgemma-4b-it-4bit)

Outputs (inside --out-dir):
  - data/{train.jsonl,valid.jsonl,test.jsonl}    (MLX-LM dataset)
  - adapters/                                    (LoRA adapters + MLX-LM README)
  - tune_config.json                             (your run settings)

Run:
  python3 tune.py --in clean.csv --prompt prompt.txt \
    --base-model mlx-community/medgemma-4b-it-4bit --out-dir medgemma-run

Then predict:
  python3 predict.py --in clean.csv --out preds.csv --prompt prompt.txt \
    --base-model mlx-community/medgemma-4b-it-4bit --adapters medgemma-run/adapters
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


REQUIRED_COLUMNS = ["split", "report", "presence", "severity", "change"]

PRESENCE_LABELS = {"present", "absent", "unknown"}
SEVERITY_LABELS = {"severe", "moderate", "mild", "trace", "unknown"}
CHANGE_LABELS = {"increased", "stable", "decreased", "unknown"}

SPLIT_MAP = {
    "training": "train",
    "train": "train",
    "validation": "valid",
    "val": "valid",
    "valid": "valid",
    "testing": "test",
    "test": "test",
}


def norm_label(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "na"
    text = str(value).strip().lower()
    if text in {"", "na", "n/a", "none", "null", "nan"}:
        return "na"
    return text


def apply_semantics(presence_value, severity_value, change_value) -> Optional[Dict[str, str]]:
    """
    Enforce your labeling semantics:
      - presence must be in PRESENCE_LABELS
      - if absent: severity/change -> na
      - if unknown: severity/change -> unknown
      - if present: allow severity/change in label sets else na
    """
    presence = norm_label(presence_value)
    severity = norm_label(severity_value)
    change = norm_label(change_value)

    if presence not in PRESENCE_LABELS:
        return None

    if presence == "absent":
        severity, change = "na", "na"
    elif presence == "unknown":
        severity, change = "unknown", "unknown"
    else:
        if severity not in SEVERITY_LABELS:
            severity = "na"
        if change not in CHANGE_LABELS:
            change = "na"

    return {"presence": presence, "severity": severity, "change": change}


def split_bucket(x: str) -> str:
    s = str(x).strip().lower()
    return SPLIT_MAP.get(s, "")


def build_chat_text(prompt_text: str, report_text: str, labels: Dict[str, str]) -> str:
    """
    We intentionally keep it dead simple and robust across templates:
      system + user + assistant(JSON)
    This becomes the single training string saved as {"text": "..."} for MLX-LM.
    """
    # A basic “Gemma-style” chat formatting works well enough for SFT here.
    # If you later want perfect model-native formatting, you can switch to
    # tokenizer.apply_chat_template — but that requires loading a tokenizer
    # before writing the dataset.
    assistant = json.dumps(labels, ensure_ascii=False)

    return (
        "<start_of_turn>system\n"
        f"{prompt_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{report_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{assistant}\n"
        "<end_of_turn>\n"
    )


def write_mlx_jsonl(df: pd.DataFrame, prompt_text: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            labels = apply_semantics(row.presence, row.severity, row.change)
            if labels is None:
                continue
            text = build_chat_text(prompt_text, str(row.report), labels)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
    return written


def run_train(
    *,
    base_model: str,
    out_dir: Path,
    iters: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    grad_checkpoint: bool,
    extra_args: list[str],
):
    data_dir = out_dir / "data"
    adapters_dir = out_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "mlx_lm.lora",
        "--model",
        base_model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapters_dir),
        "--iters",
        str(iters),
        "--learning-rate",
        str(learning_rate),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
    ]
    if grad_checkpoint:
        cmd.append("--grad-checkpoint")
    cmd.extend(extra_args)

    print("\n=== TRAIN COMMAND ===")
    print(" ".join(cmd))
    print("=====================\n")

    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser("LoRA tune a MedGemma MLX model using MLX-LM")
    p.add_argument("--in", dest="in_path", required=True, help="Clean CSV path")
    p.add_argument("--prompt", dest="prompt_path", required=True, help="Prompt/system file")
    p.add_argument("--base-model", dest="base_model", required=True, help="MLX model id/path")
    p.add_argument("--out-dir", dest="out_dir", required=True, help="Output run dir (created)")

    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-checkpoint", action="store_true")

    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-valid", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)

    # Anything after `--` is forwarded to `python -m mlx_lm.lora ...`
    p.add_argument(
        "--mlx-extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to MLX-LM (put after --mlx-extra ...)",
        default=[],
    )

    args = p.parse_args()

    in_path = Path(args.in_path)
    prompt_path = Path(args.prompt_path)
    out_dir = Path(args.out_dir)

    if not in_path.is_file():
        raise FileNotFoundError(in_path)
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "adapters").mkdir(parents=True, exist_ok=True)

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()

    df = pd.read_csv(in_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["__bucket"] = df["split"].map(split_bucket)

    train_df = df[df["__bucket"] == "train"].copy()
    valid_df = df[df["__bucket"] == "valid"].copy()
    test_df = df[df["__bucket"] == "test"].copy()

    if args.limit_train is not None:
        train_df = train_df.head(args.limit_train).copy()
    if args.limit_valid is not None:
        valid_df = valid_df.head(args.limit_valid).copy()
    if args.limit_test is not None:
        test_df = test_df.head(args.limit_test).copy()

    if len(train_df) == 0:
        raise ValueError("No training rows found. Expected split in {training,train}.")

    data_dir = out_dir / "data"
    n_train = write_mlx_jsonl(train_df, prompt_text, data_dir / "train.jsonl")
    n_valid = write_mlx_jsonl(valid_df, prompt_text, data_dir / "valid.jsonl") if len(valid_df) else 0
    n_test = write_mlx_jsonl(test_df, prompt_text, data_dir / "test.jsonl") if len(test_df) else 0

    if n_train == 0:
        raise ValueError("No usable training rows were written (semantics filtered everything).")

    print(f"\nTrain examples: {n_train}")
    print(f"Valid examples: {n_valid}")
    print(f"Test examples:  {n_test}")
    print(f"Run dir:        {out_dir}")

    # Save config for sanity/repro
    cfg = {
        "in": str(in_path),
        "prompt": str(prompt_path),
        "base_model": args.base_model,
        "out_dir": str(out_dir),
        "iters": int(args.iters),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "grad_checkpoint": bool(args.grad_checkpoint),
        "counts": {"train": n_train, "valid": n_valid, "test": n_test},
        "mlx_extra": args.mlx_extra,
    }
    (out_dir / "tune_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    run_train(
        base_model=args.base_model,
        out_dir=out_dir,
        iters=int(args.iters),
        learning_rate=float(args.lr),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        grad_checkpoint=bool(args.grad_checkpoint),
        extra_args=list(args.mlx_extra),
    )

    print("\n=== DONE ===")
    print(f"Run dir:    {out_dir}")
    print(f"Adapters:   {out_dir / 'adapters'}")
    print("\nNext (predict):")
    print(
        "  python3 predict.py --in <clean.csv> --out preds.csv "
        f"--base-model {args.base_model} --adapters {out_dir / 'adapters'} --prompt {prompt_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
