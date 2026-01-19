import sys, logging, argparse
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "Train/Test/Val",
    "Edema",
    "Radiologist_Report",
]

# Set up logging to stderr
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='[%(levelname)s] %(message)s'
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and Process CSV data"
    )

    parser.add_argument(
        "-i", "--input",
        dest="input_path",
        required=False,
        help="Input CSV file path (if omitted, read from stdin)"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        required=False,
        help="Output CSV file path (if omitted, write to stdout)"
    )

    # Parse Arguments
    args = parser.parse_args()

    # Determine input source
    if args.input_path:
        input_path = Path(args.input_path)
        logging.info(f"Reading from {input_path}")
        df = pd.read_csv(input_path)
    else:
        logging.info("Reading from stdin")
        df = pd.read_csv(sys.stdin)

    # Determine output destination
    if args.output_path:
        output_path = Path(args.output_path)
        # Ensure Output Directory Exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_func = lambda df: df.to_csv(output_path, index=False)
        logging.info(f"Writing to {output_path}")
    else:
        output_func = lambda df: df.to_csv(sys.stdout, index=False)
        logging.info("Writing to stdout")

    # Validate Expected Columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        logging.error(f"Missing required columns {missing}")
        sys.exit(1)

    # Subset Data
    df = df[REQUIRED_COLUMNS].copy()

    # Process Edema Column
    parts = df["Edema"].str.split(r"\s*\n\s*", expand=True)
    if parts.shape[1] < 3:
        logging.error(
            f"Failed to parse 'Edema' column: expected 3 parts, got {parts.shape[1]}"
        )
        sys.exit(1)
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
    output_func(df)
    logging.info("Data cleaning and processing completed successfully")
