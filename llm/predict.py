#!/usr/bin/env python3
"""
predict.py

Local inference using MLX-LM + your LoRA adapters.

Key fix:
  MLX-LM versions differ: some do NOT accept temperature/temp kwargs.
  We use make_sampler(...) and pass sampler=... into generate().

Input CSV columns:
  split,report,presence,severity,change

Output:
  Writes a new CSV with pred_presence, pred_severity, pred_change, and raw_model_output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

from mlx_lm import load, generate  # MLX-LM
from mlx_lm.sample_utils import make_sampler


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """
    Try hard to pull a JSON object out of the model output.
    Accepts:
      - pure JSON: {"presence":"present",...}
      - JSON wrapped in extra text
      - codefences
    """
    if text is None:
        return None
    s = str(text).strip()

    # Strip code fences if present
    if "```" in s:
        parts = [p.strip() for p in s.split("```") if p.strip()]
        # try the largest chunk
        s = max(parts, key=len) if parts else s

    # Find first {...} span
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def build_prompt(prompt_text: str, report_text: str) -> str:
    # Keep consistent with tune.py formatting
    return (
        "<start_of_turn>system\n"
        f"{prompt_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{report_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def main() -> int:
    p = argparse.ArgumentParser("Predict with MLX-LM + LoRA adapters")
    p.add_argument("--in", dest="in_path", required=True, help="Clean CSV path")
    p.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    p.add_argument("--prompt", dest="prompt_path", required=True, help="Prompt/system file")
    p.add_argument("--base-model", dest="base_model", required=True, help="MLX model id/path")
    p.add_argument("--adapters", dest="adapters", required=True, help="Adapters dir from tune.py")

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--temp", type=float, default=0.0)

    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    prompt_path = Path(args.prompt_path)
    adapters_dir = Path(args.adapters)

    if not in_path.is_file():
        raise FileNotFoundError(in_path)
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    if not adapters_dir.exists():
        raise FileNotFoundError(adapters_dir)

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    df = pd.read_csv(in_path)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    # Load model + adapters
    model, tokenizer = load(args.base_model, adapter_path=str(adapters_dir))

    sampler = make_sampler(
        temp=float(args.temp),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
    )

    preds_presence = []
    preds_severity = []
    preds_change = []
    raws = []

    for report in tqdm(df["report"].astype(str).tolist(), desc="Inferring"):
        prompt = build_prompt(prompt_text, report)

        # IMPORTANT: pass sampler instead of temperature kwargs
        out_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=int(args.max_tokens),
            sampler=sampler,
        )

        raws.append(out_text)

        obj = safe_json_extract(out_text)
        if obj is None:
            preds_presence.append("parse_error")
            preds_severity.append("parse_error")
            preds_change.append("parse_error")
            continue

        preds_presence.append(str(obj.get("presence", "missing")).strip().lower())
        preds_severity.append(str(obj.get("severity", "missing")).strip().lower())
        preds_change.append(str(obj.get("change", "missing")).strip().lower())

    out_df = df.copy()
    out_df["predicted_presence"] = preds_presence
    out_df["predicted_severity"] = preds_severity
    out_df["predicted_change"] = preds_change
    out_df["raw_model_output"] = raws

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
