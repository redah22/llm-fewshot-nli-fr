"""
Analyse des résultats LLM Few-Shot NLI depuis l'export WandB.
Filtre uniquement les modèles génératifs (LLMs décodeurs).

Usage:
    python results/analyze_llm_results.py
"""

import os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "label_only_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────
# MAPPING NOM DE RUN → MODÈLE LISIBLE
# ─────────────────────────────────────────────────────
LLM_PATTERNS = {
    r"gemma2_9b":        "Gemma 2 9B",
    r"mistral_7b":       "Mistral 7B",
    r"llama3_8b":        "Llama 3 8B",
    r"qwen25_7b":        "Qwen 2.5 7B",
    r"deepseek_r1_8b":   "DeepSeek R1 8B",
    r"phi3_3_8b":        "Phi-3 3.8B",
    r"mistral_nemo_12b": "Mistral NeMo 12B",
    r"mixtral_8x7b":     "Mixtral 8x7B",
    r"qwen25_14b":       "Qwen 2.5 14B",
}

MODEL_PARAMS = {
    "Phi-3 3.8B":       3.8,
    "Mistral 7B":       7.0,
    "Qwen 2.5 7B":      7.0,
    "Llama 3 8B":       8.0,
    "DeepSeek R1 8B":   8.0,
    "Gemma 2 9B":       9.0,
    "Mistral NeMo 12B": 12.0,
    "Qwen 2.5 14B":     14.0,
    "Mixtral 8x7B":     47.0,
}

# Ordre d'affichage par taille croissante
MODEL_ORDER = sorted(MODEL_PARAMS.keys(), key=lambda m: MODEL_PARAMS[m])

PALETTE = {
    "Phi-3 3.8B":       "#4CAF50",
    "Mistral 7B":       "#2196F3",
    "Qwen 2.5 7B":      "#FF9800",
    "Llama 3 8B":       "#9C27B0",
    "DeepSeek R1 8B":   "#F44336",
    "Gemma 2 9B":       "#00BCD4",
    "Mistral NeMo 12B": "#795548",
    "Qwen 2.5 14B":     "#E91E63",
    "Mixtral 8x7B":     "#607D8B",
}

def detect_model(run_name: str) -> str:
    name = run_name.lower()
    for pattern, label in LLM_PATTERNS.items():
        if re.search(pattern, name):
            return label
    return None

def extract_meta(run_name: str) -> dict:
    m = re.search(r"train-([a-z0-9\-]+)_eval-([a-z0-9\-]+)_n(\d+)_s(\d+)", run_name.lower())
    if m:
        return {
            "train_dataset": m.group(1),
            "eval_dataset":  m.group(2),
            "n_shots":       int(m.group(3)),
            "seed":          int(m.group(4)),
        }
    return None

# ─────────────────────────────────────────────────────
# 1. CHARGEMENT ET NETTOYAGE
# ─────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    raw_path = os.path.join(OUT_DIR, "wandb_export_raw.csv")
    df = pd.read_csv(raw_path)

    # Garder uniquement les runs finished
    df = df[df["State"] == "finished"].copy()

    records = []
    for _, row in df.iterrows():
        model = detect_model(str(row["Name"]))
        if model is None:
            continue
        meta = extract_meta(str(row["Name"]))
        if meta is None:
            continue

        acc = row.get("test/accuracy")
        f1  = row.get("test/f1_macro")
        pr  = row.get("test/parse_rate")

        if pd.isna(acc):
            continue

        records.append({
            "model":         model,
            "n_params_B":    MODEL_PARAMS[model],
            "train_dataset": meta["train_dataset"],
            "eval_dataset":  meta["eval_dataset"],
            "n_shots":       meta["n_shots"],
            "seed":          meta["seed"],
            "accuracy":      float(acc),
            "f1_macro":      float(f1) if not pd.isna(f1) else None,
            "parse_rate":    float(pr) if not pd.isna(pr) else None,
            "intra_cross":   "Intra" if meta["train_dataset"] == meta["eval_dataset"] else "Cross",
        })

    result = pd.DataFrame(records)
    print(f"✅ {len(result)} runs LLM chargés")
    print(result.groupby("model")["accuracy"].count().rename("nb_runs").to_string())
    return result

# ─────────────────────────────────────────────────────
# 2. GRAPHIQUES
# ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)

def plot_accuracy_vs_shots(df):
    avg = df.groupby(["model", "n_shots"])["accuracy"].mean().reset_index()
    present = [m for m in MODEL_ORDER if m in avg["model"].values]

    fig, ax = plt.subplots(figsize=(13, 6))
    for model in present:
        sub = avg[avg["model"] == model].sort_values("n_shots")
        ax.plot(sub["n_shots"], sub["accuracy"], marker="o", label=model,
                linewidth=2.2, color=PALETTE.get(model), markersize=7)

    ax.set_xlabel("Nombre de shots", fontsize=13)
    ax.set_ylabel("Accuracy moyenne (tous datasets)", fontsize=13)
    ax.set_title("Courbe Few-Shot : Performance par LLM", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1, 3, 5, 10])
    ax.set_ylim(0.3, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.axhline(1/3, color="gray", linestyle="--", linewidth=1, label="Baseline aléatoire (33%)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_accuracy_vs_shots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")

def plot_accuracy_vs_params(df):
    sub = df[df["n_shots"] == 5].groupby("model").agg(
        acc_mean=("accuracy", "mean"),
        acc_std =("accuracy", "std"),
    ).reset_index()
    sub["n_params_B"] = sub["model"].map(MODEL_PARAMS)
    sub = sub.dropna().sort_values("n_params_B")

    fig, ax = plt.subplots(figsize=(11, 6))
    for _, row in sub.iterrows():
        color = PALETTE.get(row["model"], "#333")
        ax.errorbar(row["n_params_B"], row["acc_mean"], yerr=row["acc_std"],
                    fmt="o", markersize=12, color=color, capsize=5, linewidth=2)
        ax.annotate(row["model"], (row["n_params_B"], row["acc_mean"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9, color=color)

    ax.set_xlabel("Taille du modèle (Milliards de paramètres)", fontsize=13)
    ax.set_ylabel("Accuracy moyenne (5-shot)", fontsize=13)
    ax.set_title("Scaling : Performance vs Taille du modèle (5-shot)", fontsize=15, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_accuracy_vs_params.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")

def plot_intra_vs_cross(df):
    avg = df.groupby(["model", "intra_cross"])["accuracy"].mean().reset_index()
    present = [m for m in MODEL_ORDER if m in avg["model"].values]
    avg = avg[avg["model"].isin(present)]
    avg["model"] = pd.Categorical(avg["model"], categories=present, ordered=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=avg, x="model", y="accuracy", hue="intra_cross",
                palette={"Intra": "#2196F3", "Cross": "#FF9800"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy moyenne (tous shots)", fontsize=13)
    ax.set_title("Généralisation : Intra-dataset vs Cross-dataset", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Mode d'évaluation")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_intra_vs_cross.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")

def plot_heatmap_per_model(df):
    present = [m for m in MODEL_ORDER if m in df["model"].values]
    for model in present:
        sub = df[df["model"] == model]
        pivot = sub.groupby(["train_dataset", "eval_dataset"])["accuracy"].mean().unstack(fill_value=np.nan)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                    ax=ax, linewidths=0.5, vmin=0.3, vmax=1.0,
                    annot_kws={"size": 11})
        ax.set_title(f"Cross-Dataset Heatmap — {model}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Dataset d'évaluation")
        ax.set_ylabel("Dataset d'entraînement (Few-Shot)")
        plt.tight_layout()
        fname = f"fig4_heatmap_{model.replace(' ', '_').replace('.', '').lower()}.png"
        path = os.path.join(OUT_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → {path}")

def plot_shots_per_dataset(df):
    """Accuracy par n_shots pour chaque dataset d'évaluation."""
    eval_ds = df["eval_dataset"].unique()
    fig, axes = plt.subplots(1, len(eval_ds), figsize=(5 * len(eval_ds), 5), sharey=True)
    if len(eval_ds) == 1:
        axes = [axes]

    for ax, ds in zip(axes, sorted(eval_ds)):
        sub = df[df["eval_dataset"] == ds]
        avg = sub.groupby(["model", "n_shots"])["accuracy"].mean().reset_index()
        present = [m for m in MODEL_ORDER if m in avg["model"].values]
        for model in present:
            msub = avg[avg["model"] == model].sort_values("n_shots")
            ax.plot(msub["n_shots"], msub["accuracy"], marker="o",
                    label=model, color=PALETTE.get(model), linewidth=2)
        ax.set_title(f"Eval: {ds}", fontsize=12, fontweight="bold")
        ax.set_xlabel("n_shots")
        ax.set_xticks([0, 1, 3, 5, 10])
        ax.set_ylim(0.3, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        if ax == axes[0]:
            ax.set_ylabel("Accuracy moyenne")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), fontsize=9)
    fig.suptitle("Performance Few-Shot par Dataset d'évaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_shots_per_eval_dataset.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")

# ─────────────────────────────────────────────────────
# 3. TABLE DE SYNTHÈSE
# ─────────────────────────────────────────────────────
def build_and_save_tables(df):
    # Table 1 : résumé par modèle × shots
    t1 = (df.groupby(["model", "n_shots"])
            .agg(accuracy_mean=("accuracy","mean"),
                 accuracy_std =("accuracy","std"),
                 f1_mean      =("f1_macro","mean"),
                 parse_rate   =("parse_rate","mean"),
                 n_runs       =("accuracy","count"))
            .round(4).reset_index())
    t1.to_csv(os.path.join(OUT_DIR, "table_summary_model_shots.csv"), index=False)
    print(f"  → table_summary_model_shots.csv")

    # Table 2 : résumé global par modèle (toutes configs)
    t2 = (df.groupby("model")
            .agg(accuracy_mean=("accuracy","mean"),
                 accuracy_std =("accuracy","std"),
                 f1_mean      =("f1_macro","mean"),
                 parse_rate   =("parse_rate","mean"),
                 n_runs       =("accuracy","count"),
                 n_params_B   =("n_params_B","first"))
            .round(4)
            .sort_values("accuracy_mean", ascending=False)
            .reset_index())
    t2.to_csv(os.path.join(OUT_DIR, "table_global_ranking.csv"), index=False)
    print(f"  → table_global_ranking.csv")

    # Table 3 : CSV brut nettoyé pour la réutilisation
    df.to_csv(os.path.join(OUT_DIR, "llm_results_clean.csv"), index=False)
    print(f"  → llm_results_clean.csv")

    print("\n=== CLASSEMENT GLOBAL (accuracy moyenne) ===")
    print(t2[["model","n_params_B","accuracy_mean","accuracy_std","f1_mean","n_runs"]].to_string(index=False))

# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()

    if df.empty:
        print("❌ Aucune donnée trouvée. Vérifiez wandb_export_raw.csv")
        exit(1)

    print("\n💾 Génération des tables CSV...")
    build_and_save_tables(df)

    print("\n📊 Génération des graphiques...")
    plot_accuracy_vs_shots(df)
    plot_accuracy_vs_params(df)
    plot_intra_vs_cross(df)
    plot_heatmap_per_model(df)
    plot_shots_per_dataset(df)

    print(f"\n✅ Tout est prêt dans results/label_only_plots/")
