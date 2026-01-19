import argparse, json, subprocess
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["split", "report", "presence", "severity", "change"]

PRESENCE_LABELS = {"present", "absent", "unknown"}
SEVERITY_LABELS = {"severe", "moderate", "mild", "trace", "unknown"}
CHANGE_LABELS = {"increased", "stable", "decreased", "unknown"}

ITERATIONS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
CHECKPOINT = False

def norm_label(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "na"
    text = str(value).strip().lower()
    if text in {"", "na", "n/a", "none", "null", "nan"}:
        return "na"
    return text

def apply_semantics(presence_value, severity_value, change_value):
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

def write_jsonl(df: pd.DataFrame, prompt_text: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            labels = apply_semantics(row.presence, row.severity, row.change)
            if labels is None:
                continue
            assistant = json.dumps(labels, ensure_ascii=False)
            text = (
                "<start_of_turn>system\n"
                f"{prompt_text}\n"
                "<end_of_turn>\n"
                "<start_of_turn>user\n"
                f"{str(row.report)}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
                f"{assistant}\n"
                "<end_of_turn>\n"
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA tune a MedGemma model"
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
        help="Output Directory Path"
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
    args = parser.parse_args()

    input_path = Path(args.input_path)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "data").mkdir(parents=True, exist_ok=True)
    data_dir = output_path / "data"

    (output_path / "adapters").mkdir(parents=True, exist_ok=True)
    adapters_dir = output_path / "adapters"
    
    prompt_path = Path(args.prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()

    df = pd.read_csv(input_path)
    df = df[REQUIRED_COLUMNS].copy()

    train_df = df[df["split"] == "training"].copy()
    valid_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "testing"].copy()

    n_train = write_jsonl(train_df, prompt_text, data_dir / "train.jsonl")
    n_valid = write_jsonl(valid_df, prompt_text, data_dir / "valid.jsonl")
    n_test = write_jsonl(test_df, prompt_text, data_dir / "test.jsonl")

    cmd = [
        "python",
        "-m",
        "mlx_lm.lora",
        "--model",
        args.model_id,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapters_dir),
        "--iters",
        str(ITERATIONS),
        "--learning-rate",
        str(LEARNING_RATE),
        "--batch-size",
        str(BATCH_SIZE),
    ]

    subprocess.run(cmd, check=True)
