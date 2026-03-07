import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()

# df = pd.read_csv("data/reports/tune.csv")
# df = df[df['edema'] != "unknown"]

# df.groupby('split')['edema'].value_counts()

# df["split"] = np.random.choice(
#     ["training", "validation", "testing"],
#     size=len(df),
#     p=[0.8, 0.1, 0.1],
# )

# df.to_csv("data/reports/tune.csv", index=False)

# ---

bnpp = pd.read_csv("data/bnpp/preds.csv")

binary_zero_shot = pd.read_csv("data/reports/binary_zero_shot/data/new.csv")
binary_50_50 = pd.read_csv("data/reports/binary_50_50/data/new.csv")
binary_75_25 = pd.read_csv("data/reports/binary_75_25/data/new.csv")
binary_25_75 = pd.read_csv("data/reports/binary_25_75/data/new.csv")

df1 = binary_zero_shot.merge(bnpp, on="name", how="left")
df2 = binary_50_50.merge(bnpp, on="name", how="left")
df3 = binary_75_25.merge(bnpp, on="name", how="left")
df4 = binary_25_75.merge(bnpp, on="name", how="left")

# ---


def dist(df, predicted_bnpp_log_variant=True):
    if predicted_bnpp_log_variant:
        bnpp_label = "predicted_bnpp_log"
    else:
        bnpp_label = "bnpp_log"

    x_min = float(df["bnpp_log"].min())
    x_max = float(df["bnpp_log"].max())

    x = alt.X(
        bnpp_label,
        title="BNPP Log",
        scale=alt.Scale(domain=[x_min, x_max], nice=False, zero=False),
    )
    y = alt.Y("density:Q", title="Density", scale=alt.Scale(domain=[0, 1]))

    dist_lines = (
        alt.Chart(df)
        .transform_density(
            bnpp_label,
            groupby=["predicted_edema"],
            as_=[bnpp_label, "density"],
            extent=[x_min, x_max],
            steps=300,
        )
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "predicted_edema:N",
                scale=alt.Scale(
                    domain=["present", "absent", "unknown"],
                    range=["red", "blue", "green"],
                ),
                legend=alt.Legend(title="Predicted Edema"),
            ),
        )
        .properties(width=600, height=400)
    )

    standard_threshold = float(np.log10(401))

    dist_rule = (
        alt.Chart(pd.DataFrame({bnpp_label: [standard_threshold]}))
        .mark_rule(strokeDash=[6, 6], color="black")
        .encode(x=x)
    )

    dist_label = (
        alt.Chart(
            pd.DataFrame(
                {
                    bnpp_label: [standard_threshold],
                    "density": [0.65],
                    "label": ["BNPP = 400"],
                }
            )
        )
        .mark_text(align="left", dx=6, color="black")
        .encode(
            x=x,
            y=y,
            text="label:N",
        )
    )

    return (dist_lines + dist_rule + dist_label).resolve_scale(x="shared")


dist_zero_shot = dist(df1)
dist_zero_shot_bnnp_actual = dist(df1, False)

dist_50_50 = dist(df2)
dist_50_50_bnnp_actual = dist(df2, False)

dist_75_25 = dist(df3)
dist_75_25_bnnp_actual = dist(df3, False)

dist_25_75 = dist(df4)
dist_25_75_bnnp_actual = dist(df4, False)

# ---


def roc_auc(df, standard_threshold=400):
    standard_threshold = np.log10(standard_threshold)
    df2 = df[df["predicted_edema"].isin(["present", "absent"])].copy()

    labels = (df2["predicted_edema"] == "present").to_numpy(dtype=bool)
    values = df2["predicted_bnpp_log"].to_numpy(dtype=float)
    thresholds = np.concatenate(
        [
            np.array([np.inf]),
            np.unique(values)[::-1],
            np.array([-np.inf]),
        ]
    )

    pos = max(labels.sum(), 1)
    neg = max((~labels).sum(), 1)

    tpr = np.empty(len(thresholds), dtype=float)
    fpr = np.empty(len(thresholds), dtype=float)
    for i, t in enumerate(thresholds):
        pred = values >= t
        tp = np.sum(pred & labels)
        fp = np.sum(pred & (~labels))
        tpr[i] = tp / pos
        fpr[i] = fp / neg

    roc_df = (
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
        .sort_values("fpr")
        .reset_index(drop=True)
    )

    auc = np.trapezoid(
        roc_df["tpr"].to_numpy(),
        roc_df["fpr"].to_numpy(),
    )

    roc_curve = (
        alt.Chart(roc_df)
        .mark_line(color="black")
        .encode(
            x=alt.X(
                "fpr:Q", title="False Positive Rate", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y(
                "tpr:Q", title="True Positive Rate", scale=alt.Scale(domain=[0, 1])
            ),
        )
        .properties(width=400, height=400, title=f"AUC = {auc:.4f}")
    )

    roc_line = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[6, 6], color="black")
        .encode(x="x:Q", y="y:Q")
    )

    roc_df["youden_j"] = roc_df["tpr"] - roc_df["fpr"]

    best_idx = roc_df["youden_j"].idxmax()
    best_t = roc_df.loc[best_idx, "threshold"]
    best_tpr = roc_df.loc[best_idx, "tpr"]
    best_fpr = roc_df.loc[best_idx, "fpr"]
    best_j = float(roc_df.loc[best_idx, "youden_j"])

    std_pred = values >= standard_threshold
    std_tp = np.sum(std_pred & labels)
    std_fp = np.sum(std_pred & (~labels))
    std_tpr = std_tp / pos
    std_fpr = std_fp / neg
    std_j = float(std_tpr - std_fpr)

    roc_markers_df = pd.DataFrame(
        [
            {
                "kind": "standard",
                "threshold": standard_threshold,
                "youden_j": std_j,
                "fpr": std_fpr,
                "tpr": std_tpr,
                "label": "Standard Threshold",
            },
            {
                "kind": "best",
                "threshold": best_t,
                "youden_j": best_j,
                "fpr": best_fpr,
                "tpr": best_tpr,
                "label": "Best Threshold",
            },
        ]
    )

    roc_markers = (
        alt.Chart(roc_markers_df)
        .mark_circle(size=80, opacity=1.0)
        .encode(
            x="fpr:Q",
            y="tpr:Q",
            color=alt.Color("kind:N", legend=None),
        )
    )

    roc_labels_1 = (
        alt.Chart(roc_markers_df)
        .mark_text(align="right", dx=0, dy=-24)
        .encode(
            x="fpr:Q",
            y="tpr:Q",
            text=alt.Text("label:N", title=None),
            color="kind:N",
        )
        .transform_calculate(
            label=f"datum.label + ' = ' + format(datum.threshold, '.3f')"
        )
    )

    roc_labels_2 = (
        alt.Chart(roc_markers_df)
        .mark_text(align="right", dx=0, dy=-8)
        .encode(
            x="fpr:Q",
            y="tpr:Q",
            text=alt.Text("youden_label:N", title=None),
            color="kind:N",
        )
        .transform_calculate(
            youden_label="'Youden J = ' + format(datum.youden_j, '.3f')"
        )
    )

    roc_chart = roc_curve + roc_line + roc_markers + roc_labels_1 + roc_labels_2
    best_bnpp = float((10**best_t) - 1)

    return roc_df, auc, best_j, best_t, best_bnpp, roc_chart


_, a1, j1, t1, b1, c1 = roc_auc(df1)
_, a2, j2, t2, b2, c2 = roc_auc(df2)
_, a3, j3, t3, b3, c3 = roc_auc(df3)
_, a4, j4, t4, b4, c4 = roc_auc(df4)

results = pd.DataFrame(
    {
        "ratio": ["zero-shot", "50-50", "75-25", "25-75"],
        "auc": np.array([a1, a2, a3, a4]).round(2),
        "youden_j": np.array([j1, j2, j3, j4]).round(2),
        "best_bnpp_log": np.array([t1, t2, t3, t4]).round(3),
        "best_bnpp": np.array([b1, b2, b3, b4]).round(1),
        "weighted_f1": [
            pd.read_csv(
                "data/reports/binary_zero_shot/eval/edema_variable_performance.csv",
                index_col=0,
            ).loc["weighted", "model"],
            pd.read_csv(
                "data/reports/binary_50_50/eval/edema_variable_performance.csv",
                index_col=0,
            ).loc["weighted", "model"],
            pd.read_csv(
                "data/reports/binary_75_25/eval/edema_variable_performance.csv",
                index_col=0,
            ).loc["weighted", "model"],
            pd.read_csv(
                "data/reports/binary_25_75/eval/edema_variable_performance.csv",
                index_col=0,
            ).loc["weighted", "model"],
        ],
    }
).sort_values(by="auc", ascending=False)

print(results)


# 21374
df_train = pd.read_csv('data/bnpp/train.csv')
# 2691
df_valid = pd.read_csv('data/bnpp/valid.csv')
# 2602
df_test = pd.read_csv('data/bnpp/test.csv')
# 3015
df_tune = pd.read_csv('data/reports/tune.csv')
df_tune = df_tune[df_tune.edema.ne('unknown')]
df_tune = df_tune[df_tune.split.isin(['training'])]

df_tune.value_counts('edema', normalize=True)

df_reports = pd.read_csv('data/reports/reports.csv')

df_test.merge(df_reports, on='name', how='inner')

df_test['name'].duplicated().sum()
df_reports['name'].duplicated().sum()

# ---

df = pd.read_csv("all_test.csv")

temp = pd.concat(
    [
        df[["bnpp_log"]].rename(columns={"bnpp_log": "x"}).assign(series="actual"),
        df[["predicted_bnpp_log"]]
        .rename(columns={"predicted_bnpp_log": "x"})
        .assign(series="predicted"),
    ],
    ignore_index=True,
).dropna()

bnpp_dist = (
    alt.Chart(temp)
    .transform_density(
        "x",
        groupby=["series"],
        as_=["x", "density"],
        bandwidth=0.2,
    )
    .mark_area(opacity=0.4)
    .encode(
        x=alt.X("x:Q", title="log10(BNPP+1)"),
        y=alt.Y("density:Q", title="Density", stack=None),
        color=alt.Color(
            "series:N",
            title="Series",
        ),
    )
)

temp = df[["age"]].dropna()

age_dist = (
    alt.Chart(temp)
    .transform_density(
        "age",
        as_=["age", "density"],
        bandwidth=5.0,
    )
    .mark_area(opacity=0.4)
    .encode(
        x=alt.X("age:Q", title="Age"),
        y=alt.Y("density:Q", title="Density"),
    )
)

age_bins = [20, 40, 60, 80, 100]
age_labels = ["20-39", "40-59", "60-79", "80-99"]

temp = df[["age", "bnpp_log", "predicted_bnpp_log", "predicted_edema"]].dropna()
temp["age_bin"] = pd.cut(
    temp["age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True,
    right=False,
)
temp = temp[temp["age_bin"].notna()]

step = 28
overlap = 1.0

bnpp_by_age = (
    alt.Chart(temp, height=step)
    .transform_density(
        "bnpp_log",
        as_=["x", "density"],
        groupby=["age_bin"],
        bandwidth=0.2,
    )
    .mark_area(opacity=0.4)
    .encode(
        x=alt.X("x:Q", title="log10(BNPP+1)"),
        y=alt.Y(
            "density:Q",
            stack=None,
            title="Density",
            axis=None,
            scale=alt.Scale(range=[step, -step * overlap]),
        ),
    )
    .facet(
        row=alt.Row(
            "age_bin:N",
            sort=age_labels,
            title=None,
            header=alt.Header(
                labelAngle=0,
                labelAlign="left",
            ),
        )
    )
    .configure_facet(spacing=6)
    .configure_view(stroke=None)
)

predicted_bnpp_by_age = (
    alt.Chart(temp, height=step)
    .transform_density(
        "predicted_bnpp_log",
        as_=["x", "density"],
        groupby=["age_bin"],
        bandwidth=0.2,
    )
    .mark_area(opacity=0.4)
    .encode(
        x=alt.X("x:Q", title="log10(BNPP+1)"),
        y=alt.Y(
            "density:Q",
            stack=None,
            title="Density",
            axis=None,
            scale=alt.Scale(range=[step, -step * overlap]),
        ),
    )
    .facet(
        row=alt.Row(
            "age_bin:N",
            sort=age_labels,
            title=None,
            header=alt.Header(
                labelAngle=0,
                labelAlign="left",
            ),
        )
    )
    .configure_facet(spacing=6)
    .configure_view(stroke=None)
)

bnpp_by_age_and_edema = (
    alt.Chart(temp, height=step)
    .transform_density(
        "bnpp_log",
        as_=["x", "density"],
        groupby=["age_bin", "predicted_edema"],
        bandwidth=0.2,
    )
    .mark_area(opacity=0.4)
    .encode(
        x=alt.X("x:Q", title="log10(BNPP+1)"),
        y=alt.Y(
            "density:Q",
            stack=None,
            axis=None,
            scale=alt.Scale(range=[step, -step * overlap]),
        ),
        color=alt.Color(
            "predicted_edema:N",
            title="Predicted edema",
        ),
    )
    .facet(
        row=alt.Row(
            "age_bin:N",
            sort=age_labels,
            title=None,
            header=alt.Header(
                labelAngle=0,
                labelAlign="left",
            ),
        ),
    )
    .configure_facet(spacing=6)
    .configure_view(stroke=None)
)

predicted_bnpp_by_age_and_edema = (
    alt.Chart(temp, height=step)
    .transform_density(
        "predicted_bnpp_log",
        as_=["x", "density"],
        groupby=["age_bin", "predicted_edema"],
        bandwidth=0.2,
    )
    .mark_area(opacity=0.4)
    .encode(
        x=alt.X("x:Q", title="log10(BNPP+1)"),
        y=alt.Y(
            "density:Q",
            stack=None,
            axis=None,
            scale=alt.Scale(range=[step, -step * overlap]),
        ),
        color=alt.Color(
            "predicted_edema:N",
            title="Predicted edema",
        ),
    )
    .facet(
        row=alt.Row(
            "age_bin:N",
            sort=age_labels,
            title=None,
            header=alt.Header(
                labelAngle=0,
                labelAlign="left",
            ),
        ),
    )
    .configure_facet(spacing=6)
    .configure_view(stroke=None)
)

df = pd.read_csv("final_results.csv")

df.predicted_edema.value_counts(normalize=True)