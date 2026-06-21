
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plot_df = pd.read_csv("quartile_binned_files_viewed_stats.csv")

dataset_order = ["LCA", "SWE"]
retriever_order = ["Qwen 3", "CodeXEmbed"]
repr_order = ["Path", "Sum", "Raw", "Bug"]

parts = plot_df["short_label"].str.split(" • ", expand=True)
plot_df["retriever_short"] = parts[0]
plot_df["repr_short"] = parts[1]

plot_df["dataset"] = pd.Categorical(plot_df["dataset"], categories=dataset_order, ordered=True)
plot_df["retriever_short"] = pd.Categorical(plot_df["retriever_short"], categories=retriever_order, ordered=True)
plot_df["repr_short"] = pd.Categorical(plot_df["repr_short"], categories=repr_order, ordered=True)

plot_df = plot_df.sort_values(["dataset", "retriever_short", "repr_short"]).reset_index(drop=True)

y_lca = np.arange(15, 7, -1)
y_swe = np.arange(6, -2, -1)
y_positions = np.concatenate([y_lca, y_swe])

bin_order = ["0 files", "Q1: 1–5", "Q2: 6–10", "Q3: 11–15", "Q4: 16–20"]
bin_colors = {
    "0 files":   "#8da0cb",
    "Q1: 1–5":   "#fc8d62",
    "Q2: 6–10":  "#66c2a5",
    "Q3: 11–15": "#e78ac3",
    "Q4: 16–20": "#a6d854",
}

fig, ax = plt.subplots(figsize=(9.4, 7.2))

left = np.zeros(len(plot_df))
for bin_name in bin_order:
    values = plot_df[bin_name].to_numpy()
    ax.barh(
        y_positions,
        values,
        left=left,
        height=0.72,
        label=bin_name,
        color=bin_colors[bin_name],
        edgecolor="none",
    )
    for y, v, lft in zip(y_positions, values, left):
        if v >= 6:
            ax.text(lft + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=7.5)
    left += values

for idx, row in plot_df.iterrows():
    q1 = row["Q1: 1–5"]
    is_non_raw = row["repr_short"] != "Raw"
    if is_non_raw and q1 > 0:
        label = f"{q1:.1f}% Q1"
        ax.annotate(
            label,
            xy=(q1, y_positions[idx]),
            xytext=(5.2, y_positions[idx]),
            textcoords="data",
            va="center",
            ha="left",
            fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.6, shrinkA=0, shrinkB=0),
        )

ax.set_xlim(0, 100)
ax.set_xlabel("Percentage of instances")
ax.set_yticks(y_positions)
ax.set_yticklabels(plot_df["short_label"])
ax.set_title("Files viewed by the ranker across representations")

ax.axhline(7.0, linewidth=1.0, color="black")

ax.text(-0.25, np.mean(y_lca), "LCA", transform=ax.get_yaxis_transform(),
        rotation=90, ha="center", va="center",
        fontsize=12, fontweight="bold", clip_on=False)
ax.text(-0.25, np.mean(y_swe), "SWE", transform=ax.get_yaxis_transform(),
        rotation=90, ha="center", va="center",
        fontsize=12, fontweight="bold", clip_on=False)

ax.grid(axis="x", linestyle=":", linewidth=0.6)
ax.set_axisbelow(True)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

fig.subplots_adjust(left=0.24, right=0.98, top=0.92, bottom=0.16)

png_path = "combined_allrepr_quartile_binned_files_viewed_grouped_annotated.png"
pdf_path = "combined_allrepr_quartile_binned_files_viewed_grouped_annotated.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.show()
