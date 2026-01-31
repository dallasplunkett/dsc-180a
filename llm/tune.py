import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

EDEMA_LABELS = ["present", "absent", "unknown"]


def write_jsonl(df, prompt_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            labels = {
                "edema": row.edema,
                "severity": row.severity,
            }

            prompt = prompt_text.replace("{report_text}", str(row.report).strip())

            if not prompt.endswith("\n"):
                prompt += "\n"

            text = prompt + json.dumps(labels, ensure_ascii=False) + "<end_of_turn>"

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1

    return written


def parse_ratio(ratio_text):
    if ratio_text is None:
        return None

    parts = [p.strip() for p in ratio_text.split("-") if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            "Ratio must be in format present-absent-unknown (e.g. 30-40-30)."
        )

    values = [int(p) for p in parts]
    if any(v < 0 for v in values):
        raise ValueError("Ratio values must be non-negative.")
    if sum(values) != 100:
        raise ValueError("Ratio values must sum to 100.")

    return dict(zip(EDEMA_LABELS, values))


def apply_ratio(df, ratio):
    if ratio is None:
        return df

    total = len(df)
    if total == 0:
        return df

    samples = []
    chosen_idx = set()

    for label in EDEMA_LABELS:
        target = int(round(total * (ratio[label] / 100)))
        subset = df[df["edema"] == label]
        take = min(len(subset), target)
        if take > 0:
            picked = subset.sample(n=take, replace=False)
            samples.append(picked)
            chosen_idx.update(picked.index.tolist())

    remaining = total - len(chosen_idx)
    if remaining > 0:
        pool = df.drop(index=list(chosen_idx))
        take = min(len(pool), remaining)
        if take > 0:
            samples.append(pool.sample(n=take, replace=False))

    if not samples:
        return df.iloc[0:0]

    return pd.concat(samples).sample(frac=1, replace=False).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA tune a MedGemma model")
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
        help="Output Directory Path",
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
        "--iters",
        type=int,
        default=200,
        help="Number of training iterations (default: 200)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--ratio",
        dest="ratio",
        default=None,
        help="Edema ratio for training split as present-absent-unknown (e.g. 30-40-30)",
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
    df = df[["split", "report", "edema", "severity"]].copy()

    train_df = df[df["split"] == "training"].copy()
    train_df = apply_ratio(train_df, parse_ratio(args.ratio))
    n_train = write_jsonl(train_df, prompt_text, data_dir / "train.jsonl")

    valid_df = df[df["split"] == "validation"].copy()
    n_valid = write_jsonl(valid_df, prompt_text, data_dir / "valid.jsonl")

    test_df = df[df["split"] == "testing"].copy()
    n_test = write_jsonl(test_df, prompt_text, data_dir / "test.jsonl")

    cmd = [
        "python",
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        args.model_id,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapters_dir),
        "--iters",
        str(args.iters),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
    ]

    subprocess.run(cmd, check=True)
