import sys, logging, argparse, json, subprocess
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='[%(levelname)s] %(message)s'
)

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

            text = (
                "<start_of_turn>system\n"
                f"{prompt_text}\n"
                "<end_of_turn>\n"
                "<start_of_turn>user\n"
                f"{str(row.report)}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
                f"{json.dumps(labels, ensure_ascii=False)}\n"
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
    logging.info(f"Reading input from {input_path}")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing output to {output_path}")

    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Creating a directory at {data_dir}")

    adapters_dir = output_path / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Creating a directory at {adapters_dir}")

    prompt_path = Path(args.prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    logging.info(f"Reading prompt from {prompt_path}")

    df = pd.read_csv(input_path)
    df = df[["split", "report", "presence", "severity", "change"]].copy()
    logging.info(f"Loaded {len(df)} rows from {input_path}")

    train_df = df[df["split"] == "training"].copy()
    n_train = write_jsonl(train_df, prompt_text, data_dir / "train.jsonl")
    logging.info(f"Wrote {n_train} training samples to {data_dir / 'train.jsonl'}")

    valid_df = df[df["split"] == "validation"].copy()
    n_valid = write_jsonl(valid_df, prompt_text, data_dir / "valid.jsonl")
    logging.info(f"Wrote {n_valid} validation samples to {data_dir / 'valid.jsonl'}")

    test_df = df[df["split"] == "testing"].copy()
    n_test = write_jsonl(test_df, prompt_text, data_dir / "test.jsonl")
    logging.info(f"Wrote {n_test} test samples to {data_dir / 'test.jsonl'}")

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
        str(200),
        "--learning-rate",
        str(1e-4),
        "--batch-size",
        str(4),
    ]

    logging.info(f"Starting fine-tuning...")
    subprocess.run(cmd, check=True)
    logging.info("Fine-tuning completed successfully")
