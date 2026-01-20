import argparse
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and Process CSV data"
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
    args = parser.parse_args()

    input_path = Path(args.input_path)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df[["Train/Test/Val", "Edema", "Radiologist_Report"]].copy()

    parts = df["Edema"].str.split(r"\s*\n\s*", expand=True)
    df["presence"] = (
        parts[0]
        .str.replace("Presence:", "", regex=False)
        .str.strip()
    )
    df["severity"] = (
        parts[1]
        .str.replace("Severity:", "", regex=False)
        .str.strip()
    )
    df["change"] = (
        parts[2]
        .str.replace("Change:", "", regex=False)
        .str.strip()
    )
    df = df.drop(columns=["Edema"])

    df = df.rename(columns={
        "Train/Test/Val": "split",
        "Radiologist_Report": "report",
    })

    for col in ["presence", "severity", "change"]:
        df[col] = (
            df[col]
            .str.lower()
            .str.strip()
            .replace(["na", ""], pd.NA)
        )

    df = df[df["presence"] != "none"].copy()

    df.to_csv(output_path, index=False)
