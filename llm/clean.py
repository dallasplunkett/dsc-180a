import argparse
import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = [
    "Train/Test/Val",
    "Edema",
    "Radiologist_Report",
]

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Clean Radiology Report")
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input CSV path"
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output CSV path"
    )

    # Parse Arguments
    args = parser.parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    # Ensure Input File Exists
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    
    # Ensure Output Directory Exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load Data
    df = pd.read_csv(in_path)

    # Validate Expected Columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Subset Data
    df = df[REQUIRED_COLUMNS].copy()

    # Process Edema Column
    parts = df["Edema"].str.split(r"\s*\n\s*", expand=True)
    if parts.shape[1] < 3:
        raise ValueError(
            f"Failed to parse 'Edema' column: expected 3 parts, got {parts.shape[1]}"
        )
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

    # Rename Columns
    df = df.rename(columns={
        "Train/Test/Val": "split",
        "Radiologist_Report": "report",
    })

    # Normalize
    for col in ["presence", "severity", "change"]:
        df[col] = (
            df[col]
            .str.lower()
            .str.strip()
            .replace(["na", ""], pd.NA)
        )

    # Drop Specific Invalid Rows
    df = df[df["presence"] != "none"].copy()

    # Export Cleaned Data
    df.to_csv(out_path, index=False)
