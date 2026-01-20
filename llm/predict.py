import sys, logging, argparse, json, re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='[%(levelname)s] %(message)s'
)

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
        description="Predict the presence, severity, and change of Edema"
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
    logging.info(f"Reading input from {input_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing output to {output_path}")

    prompt_path = Path(args.prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    logging.info(f"Reading prompt from {prompt_path}")

    adapter_path = None
    if args.adapters_path:
        adapter_path = str(Path(args.adapters_path))
        logging.info(f"Loading model {args.model_id} with adapters from {adapter_path}")
    else:
        logging.info(f"Loading model {args.model_id}")
    
    df = pd.read_csv(input_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()
        logging.info(f"Input limited to first {args.limit} rows")

    model, tokenizer = load(args.model_id, adapter_path=adapter_path)
    logging.info(f"Loading model {args.model_id} with adapters from {adapter_path}")

    predicted_presence = []
    predicted_severity = []
    predicted_change = []

    raw_outputs = []

    sampler = make_sampler(0.0)

    logging.info(f"Processing {len(df)} reports")
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
            predicted_presence.append("parse_error")
            predicted_severity.append("parse_error")
            predicted_change.append("parse_error")
            continue

        # Normalize keys
        obj = {str(k).strip().lower(): v for k, v in obj.items()}

        p = str(obj.get("presence", "missing")).strip().lower()
        s = str(obj.get("severity", "missing")).strip().lower()
        c = str(obj.get("change", "missing")).strip().lower()

        # Enforce semantic dependency rule
        if p != "present":
            s = "na"
            c = "na"

        if p == "present":
            if s == "na":
                s = "unknown"
            if c == "na":
                c = "unknown"

        predicted_presence.append(p)
        predicted_severity.append(s)
        predicted_change.append(c)

    out_df = df.copy()

    out_df["predicted_presence"] = predicted_presence
    out_df["predicted_severity"] = predicted_severity
    out_df["predicted_change"] = predicted_change

    out_df["raw_model_output"] = raw_outputs

    out_df.to_csv(output_path, index=False)
    logging.info("Prediction completed successfully")
