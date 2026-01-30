import numpy as np
import pandas as pd
import altair as alt
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
            alt.datum.norm > 0.5,
            alt.value("white"),
            alt.value("black")
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

    return pd.DataFrame({
        label: {
            "support": int(row_sums[i]),
            "precision": round(float(precision[i]), 2),
            "recall": round(float(recall[i]), 2),
        }
        for i, label in enumerate(labels)
    })

def evaluate_variable(y_true, y_pred, labels, iters=2000, confidence=0.95):
    metrics = ["micro", "macro", "weighted"]

    model = np.array(
        [f1_score(y_true, y_pred, labels=labels, average=metric) for metric in metrics], # type: ignore
        dtype=float,
    )

    counts = pd.Series(y_true).value_counts()
    probs = np.array([counts.get(label, 0) for label in labels], dtype=float)
    probs = probs / probs.sum()

    y_true_np = np.asarray(y_true, dtype=object)
    sims = np.empty((iters, len(metrics)), dtype=float)

    for i in range(iters):
        y_rand = np.random.choice(labels, size=len(y_true_np), replace=True, p=probs)
        sims[i] = [f1_score(y_true_np, y_rand, labels=labels, average=metric) for metric in metrics] # type: ignore

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

if __name__ == "__main__":
    # Load Data
    df = pd.read_csv("../data/reports/zero_shot_predictions.csv")
    df = df[df['split'] == 'testing']
    df = df[df["edema"].isin(EDEMA_LABELS)].copy()

    # Build Confusion Matrices
    edema_count_cm = pd.DataFrame(
        confusion_matrix(
            df["edema"],
            df["predicted_edema"],
            labels=EDEMA_LABELS,
        ),
        index=pd.Index(EDEMA_LABELS, name="actual"),
        columns=pd.Index(EDEMA_LABELS, name="predicted"),
    )
    edema_norm_cm = pd.DataFrame(
        confusion_matrix(
            df["edema"],
            df["predicted_edema"],
            labels=EDEMA_LABELS,
            normalize="true",
        ),
        index=pd.Index(EDEMA_LABELS, name="actual"),
        columns=pd.Index(EDEMA_LABELS, name="predicted"),
    )
    actual_present = df["edema"] == "present"

    severity_mask = (
        actual_present
        & df["severity"].notna()
        & df["predicted_severity"].notna()
    )
    severity_count_cm = pd.DataFrame(
        confusion_matrix(
            df.loc[severity_mask, "severity"],
            df.loc[severity_mask, "predicted_severity"],
            labels=SEVERITY_LABELS,
        ),
        index=pd.Index(SEVERITY_LABELS, name="actual"),
        columns=pd.Index(SEVERITY_LABELS, name="predicted"),
    )
    severity_norm_cm = pd.DataFrame(
        confusion_matrix(
            df.loc[severity_mask, "severity"],
            df.loc[severity_mask, "predicted_severity"],
            labels=SEVERITY_LABELS,
            normalize="true",
        ),
        index=pd.Index(SEVERITY_LABELS, name="actual"),
        columns=pd.Index(SEVERITY_LABELS, name="predicted"),
    )

    # Build Confusion Matrix Chart
    edema_chart = create_cm_chart(
        edema_norm_cm,
        edema_count_cm,
        "Edema Confusion Matrix",
        EDEMA_LABELS,
    )
    severity_chart = create_cm_chart(
        severity_norm_cm,
        severity_count_cm,
        "Severity Confusion Matrix",
        SEVERITY_LABELS,
    )

    (edema_chart & severity_chart).show()


    # Performance Evaluation Tables
    edema_category_performance = evaluate_categories(
        df["edema"],
        df["predicted_edema"],
        EDEMA_LABELS,
    )
    edema_variable_performance = evaluate_variable(
        df["edema"],
        df["predicted_edema"],
        EDEMA_LABELS,
    )

    severity_category_performance = evaluate_categories(
        df.loc[severity_mask, "severity"],
        df.loc[severity_mask, "predicted_severity"],
        SEVERITY_LABELS,
    )
    severity_variable_performance = evaluate_variable(
        df.loc[severity_mask, "severity"],
        df.loc[severity_mask, "predicted_severity"],
        SEVERITY_LABELS,
    )
