import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams["font.family"] = "sans-serif"
plt.style.use("dark_background")

CSV_PATH    = "results/metrics/few_shot_sweep_cross_fracas_gqnli.csv"
OUTPUT_DIR  = "results/metricsgraphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df[df["State"] == "finished"].dropna(subset=["n_shots"]).copy()
df["acc"] = df["final_test_accuracy"].combine_first(df["test/accuracy"])
df["f1"]  = df["final_test_f1_score"].combine_first(df["test/f1_score"])

agg = df.groupby(["model_key", "peft_method", "n_shots"]).agg(
    acc_mean=("acc", "mean"), acc_std=("acc", "std"),
    f1_mean=("f1", "mean"),  f1_std=("f1", "std"),
).reset_index()

STYLES = {
    ("camembert", "lora"): {"color": "#4FC3F7", "marker": "o", "label": "CamemBERT + LoRA"},
    ("camembert", "ia3"):  {"color": "#81C784", "marker": "s", "label": "CamemBERT + IA³"}
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Few-Shot Cross-Dataset : FraCaS → GQNLI-FR\n(moyenne sur 3 seeds, barres = ±1 std)",
             fontsize=14, fontweight="bold", color="white", y=1.02)

for ax, metric, ylabel in zip(axes, ["acc", "f1"], ["Accuracy", "F1-score macro"]):
    for (model, peft), style in STYLES.items():
        sub = agg[(agg.model_key == model) & (agg.peft_method == peft)].sort_values("n_shots")
        if sub.empty:
            continue
        ax.errorbar(
            sub["n_shots"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"].fillna(0),
            label=style["label"], color=style["color"],
            marker=style["marker"], markersize=7, linewidth=2, capsize=4, capthick=1.5
        )
    ax.axhline(1/3, color="white", linestyle="--", alpha=0.4, linewidth=1, label="Hasard (33%)")
    ax.set_xlabel("Nombre d'exemples few-shot (n_shots) issus de FraCaS", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12)
    ax.set_xticks([8, 16, 32, 64, 128])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_ylim(0, 1.0)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "few_shot_cross_fracas_gqnli.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"✅ Graphique sauvegardé : {out_path}")
