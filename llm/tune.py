import argparse, json, subprocess
from pathlib import Path
import pandas as pd

def write_jsonl(df, prompt_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            labels = {
                "presence": row.presence,
                "severity": row.severity,
                "change": row.change,
            }

            prompt = prompt_text.replace("{report_text}", str(row.report).strip())

            if not prompt.endswith("\n"):
                prompt += "\n"

            text = prompt + json.dumps(labels, ensure_ascii=False) + "<end_of_turn>"

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
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of training iterations (default: 200)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    adapters_dir = output_path / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()

    df = pd.read_csv(input_path)
    df = df[["split", "report", "presence", "severity", "change"]].copy()

    train_df = df[df["split"] == "training"].copy()
    n_train = write_jsonl(train_df, prompt_text, data_dir / "train.jsonl")

    valid_df = df[df["split"] == "validation"].copy()
    n_valid = write_jsonl(valid_df, prompt_text, data_dir / "valid.jsonl")

    test_df = df[df["split"] == "testing"].copy()
    n_test = write_jsonl(test_df, prompt_text, data_dir / "test.jsonl")

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", args.model_id,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapters_dir),
        "--iters", str(args.iters),
        "--learning-rate", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
    ]

    subprocess.run(cmd, check=True)
