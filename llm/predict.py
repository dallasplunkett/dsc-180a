import argparse
import json
import re
from pathlib import Path

import pandas as pd
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

EDEMA_LABELS = {"present", "absent"}
SEVERITY_LABELS = {"severe", "moderate", "mild", "trace", "na"}


def parse_raw_model_output(raw_text):
    fallback = {"edema": "na", "severity": "na"}
    if raw_text is None:
        return fallback

    raw = str(raw_text).strip().replace("<end_of_turn>", "").strip()
    if not raw:
        return fallback

    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE
    )
    candidate = match.group(1).strip() if match else raw
    if '""' in candidate:
        candidate = candidate.replace('""', '"')

    try:
        obj = json.loads(candidate)
    except Exception:
        return fallback

    if not isinstance(obj, dict):
        return fallback

    normalized = {
        str(k).strip().lower(): str(v).strip().lower() for k, v in obj.items()
    }

    edema = normalized.get("edema", "na")
    severity = normalized.get("severity", "na")

    normalized["edema"] = edema if edema in EDEMA_LABELS else "na"
    normalized["severity"] = severity if severity in SEVERITY_LABELS else "na"

    return normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict edema and severity")
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "-p", "--prompt", dest="prompt_path", required=True, help="Input TXT file path"
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_id",
        required=True,
        help="HuggingFace model ID (see https://huggingface.co/models?library=mlx for options)",
    )
    parser.add_argument(
        "-a",
        "--adapters",
        dest="adapters_path",
        default=None,
        help="Path to adapters directory (optional). If omitted, runs base model.",
    )
    parser.add_argument("-l", "--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_path)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()

    adapter_path = None
    if args.adapters_path:
        adapter_path = str(Path(args.adapters_path))

    df = pd.read_csv(input_path)
    if "edema" in df.columns:
        df = df[df["edema"] != "unknown"]
    if args.limit is not None:
        df = df.head(args.limit).copy()

    model, tokenizer = load(args.model_id, adapter_path=adapter_path)  # type: ignore

    predicted_edema = []
    predicted_severity = []

    raw_model_output = []

    sampler = make_sampler(0.0)

    for report in tqdm(df["report"].astype(str).tolist(), desc="Inferring"):
        prompt = prompt_text.replace("{report_text}", report.strip())

        raw = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=128,
            sampler=sampler,
        )
        raw_model_output.append(raw)

        predictions = parse_raw_model_output(raw)
        predicted_edema.append(predictions["edema"])
        predicted_severity.append(predictions["severity"])

    out_df = df.copy()

    out_df["predicted_edema"] = predicted_edema
    out_df["predicted_severity"] = predicted_severity

    out_df["raw_model_output"] = raw_model_output

    out_df.to_csv(output_path, index=False)
