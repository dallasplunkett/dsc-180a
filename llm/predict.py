import argparse, json, re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def extract_json(text):
    if text is None:
        return None

    s = str(text).strip()
    s = s.replace("<end_of_turn>", "").strip()

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()

    try:
        return json.loads(s)
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict edema and severity"
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_path",
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "-p", "--prompt",
        dest="prompt_path",
        required=True,
        help="Input TXT file path"
    )
    parser.add_argument(
        "-m", "--model",
        dest="model_id",
        required=True,
        help="HuggingFace model ID (see https://huggingface.co/models?library=mlx for options)"
    )
    parser.add_argument(
        "-a", "--adapters",
        dest="adapters_path",
        default=None,
        help="Path to adapters directory (optional). If omitted, runs base model."
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None
    )
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
    if args.limit is not None:
        df = df.head(args.limit).copy()

    model, tokenizer = load(args.model_id, adapter_path=adapter_path) # type: ignore

    predicted_edema = []
    predicted_severity = []

    raw_outputs = []

    sampler = make_sampler(0.0)

    for report in tqdm(df["report"].astype(str).tolist(), desc="Inferring"):
        prompt = prompt_text.replace("{report_text}", report.strip())
        if not prompt.endswith("\n"):
            prompt += "\n"

        out_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=128,
            sampler=sampler,
        )

        raw = str(out_text)
        raw_outputs.append(raw)

        obj = extract_json(raw)

        if obj is None:
            predicted_edema.append("parse_error")
            predicted_severity.append("parse_error")
            continue

        obj = {str(k).strip().lower(): v for k, v in obj.items()}

        p = str(obj.get("edema", "missing")).strip().lower()
        s = str(obj.get("severity", "missing")).strip().lower()

        if p != "present":
            s = "na"

        if p == "present" and s == "na":
            s = "unknown"

        predicted_edema.append(p)
        predicted_severity.append(s)

    out_df = df.copy()

    out_df["predicted_edema"] = predicted_edema
    out_df["predicted_severity"] = predicted_severity

    out_df["raw_model_output"] = raw_outputs

    out_df.to_csv(output_path, index=False)
