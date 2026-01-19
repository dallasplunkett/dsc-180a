import sys, logging, argparse, json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from mlx_lm import load, generate

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='[%(levelname)s] %(message)s'
)

MAX_TOKENS = 128
TOP_P = 0.95
TOP_K = 64
TEMP = 0.0

def extract_json(text):
    if text is None:
        return None
    s = str(text).strip()

    if "```" in s:
        parts = [p.strip() for p in s.split("```") if p.strip()]
        s = max(parts, key=len) if parts else s

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
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
        dest="out_path",
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
        required=True,
        help="Path to adapters directory"
    )
    parser.add_argument(
        "l", "--limit",
        type=int,
        default=None
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    logging.info(f"Reading input from {input_path}")

    out_path = Path(args.out_path)
    logging.info(f"Writing output to {out_path}")

    prompt_path = Path(args.prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    logging.info(f"Reading prompt from {prompt_path}")

    adapters_path = str(Path(args.adapters_path))
    
    df = pd.read_csv(input_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()
        logging.info(f"Input limited to first {args.limit} rows")

    model, tokenizer = load(args.model_id, adapter_path=adapters_path)
    logging.info(f"Loading model {args.model_id} with adapters from {adapters_path}")

    predicted_presence = []
    predicted_severity = []
    predicted_change = []
    raws = []

    logging.info(f"Processing {len(df)} reports")
    for report in tqdm(df["report"].astype(str).tolist(), desc="Inferring"):
        prompt = (
            "<start_of_turn>system\n"
            f"{prompt_text}\n"
            "<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"{report}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        out_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temp=TEMP,
            top_p=TOP_P,
            top_k=TOP_K,
        )

        raws.append(out_text)

        obj = extract_json(out_text)
        if obj is None:
            predicted_presence.append("parse_error")
            predicted_severity.append("parse_error")
            predicted_change.append("parse_error")
            continue

        predicted_presence.append(str(obj.get("presence", "missing")).strip().lower())
        predicted_severity.append(str(obj.get("severity", "missing")).strip().lower())
        predicted_change.append(str(obj.get("change", "missing")).strip().lower())

    out_df = df.copy()
    out_df["predicted_presence"] = predicted_presence
    out_df["predicted_severity"] = predicted_severity
    out_df["predicted_change"] = predicted_change
    out_df["raw_model_output"] = raws

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logging.info(f"Prediction completed successfully")
