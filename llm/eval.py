import argparse
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

EDEMA_LABELS = ["present", "absent", "unknown"]
SEVERITY_LABELS = ["severe", "moderate", "mild", "trace", "unknown"]


def create_cm_chart(norm_cm, count_cm, title, labels):
    norm = norm_cm.reset_index().melt(
        id_vars="actual", var_name="predicted", value_name="norm"
    )
    count = count_cm.reset_index().melt(
        id_vars="actual", var_name="predicted", value_name="count"
    )
    data = norm.merge(count, on=["actual", "predicted"], how="left")
    data["label"] = data.apply(lambda r: f"{r['norm']:.2f} ({int(r['count'])})", axis=1)

    base = alt.Chart(data).encode(
        x=alt.X("predicted:N", sort=labels),
        y=alt.Y("actual:N", sort=labels),
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color("norm:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            "actual:N",
            "predicted:N",
            alt.Tooltip("norm:Q", format=".2f"),
            alt.Tooltip("count:Q", format="d"),
        ],
    )

    text = base.mark_text(baseline="middle").encode(
        text="label:N",
        color=alt.condition(
            alt.datum.norm > 0.5, alt.value("white"), alt.value("black")
        ),
    )

    return (heatmap + text).properties(width=400, height=400, title=title)


def safe_division(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.zeros(np.broadcast(num, den).shape, dtype=float)
    np.divide(num, den, out=out, where=(den != 0))
    return out


def evaluate_categories(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    diag = np.diag(cm)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)

    precision = safe_division(diag, diag + (col_sums - diag))
    recall = safe_division(diag, row_sums)

    return pd.DataFrame(
        {
            label: {
                "support": int(row_sums[i]),
                "precision": round(float(precision[i]), 2),
                "recall": round(float(recall[i]), 2),
            }
            for i, label in enumerate(labels)
        }
    )


def evaluate_variable(y_true, y_pred, labels, iters=2000, confidence=0.95):
    metrics = ["micro", "macro", "weighted"]

    model = np.array(
        [
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average=metric,
                zero_division=0,
            )
            for metric in metrics
        ],
        dtype=float,
    )

    counts = pd.Series(y_true).value_counts()
    probs = np.array([counts.get(label, 0) for label in labels], dtype=float)
    probs = probs / probs.sum()

    y_true_np = np.asarray(y_true, dtype=object)
    sims = np.empty((iters, len(metrics)), dtype=float)

    for i in range(iters):
        y_rand = np.random.choice(labels, size=len(y_true_np), replace=True, p=probs)
        sims[i] = [
            f1_score(
                y_true_np,
                y_rand,
                labels=labels,
                average=metric,
                zero_division=0,
            )
            for metric in metrics
        ]

    quantile_low = (1.0 - confidence) / 2.0
    quantile_high = 1.0 - quantile_low

    rand_mean = sims.mean(axis=0)
    rand_low = np.quantile(sims, quantile_low, axis=0)
    rand_high = np.quantile(sims, quantile_high, axis=0)

    lift_sims = safe_division(model[None, :], sims)
    lift_mean = lift_sims.mean(axis=0)

    out = pd.DataFrame(
        {
            "model": np.round(model, 2),
            "low": np.round(rand_low, 2),
            "random": np.round(rand_mean, 2),
            "high": np.round(rand_high, 2),
            "lift": np.round(lift_mean, 2),
        },
        index=pd.Index(metrics),
    )

    return out


def save_table(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def log_table(title, df):
    print(f"\n{title}")
    print(df.to_string())


def log_link(path, label):
    print(f"{label}: {path.resolve().as_uri()}")


def evaluate_target(
    df,
    y_true_col,
    y_pred_col,
    labels,
    title,
    name,
    output_path,
):
    count_cm = pd.DataFrame(
        confusion_matrix(
            df[y_true_col],
            df[y_pred_col],
            labels=labels,
        ),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    norm_cm = pd.DataFrame(
        confusion_matrix(
            df[y_true_col],
            df[y_pred_col],
            labels=labels,
            normalize="true",
        ),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )

    chart = create_cm_chart(norm_cm, count_cm, title, labels)
    category_perf = evaluate_categories(df[y_true_col], df[y_pred_col], labels)
    variable_perf = evaluate_variable(df[y_true_col], df[y_pred_col], labels)

    log_table(f"{name} category performance", category_perf)
    log_table(f"{name} variable performance", variable_perf)

    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / f"{name.lower()}_confusion_matrix.html"
    chart.save(chart_path)
    log_link(chart_path, f"{name} confusion matrix")
    save_table(category_perf, output_path / f"{name.lower()}_category_performance.csv")
    save_table(variable_perf, output_path / f"{name.lower()}_variable_performance.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate edema/severity predictions")
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
        help="Directory to write charts and tables",
    )
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input_path))
    df = df[df["split"] == "testing"].copy()

    output_path = Path(args.output_path)

    evaluate_target(
        df,
        "edema",
        "predicted_edema",
        EDEMA_LABELS,
        "Edema Confusion Matrix",
        "Edema",
        output_path,
    )

    severity_df = df[df["edema"] == "present"]
    evaluate_target(
        severity_df,
        "severity",
        "predicted_severity",
        SEVERITY_LABELS,
        "Severity Confusion Matrix",
        "Severity",
        output_path,
    )
