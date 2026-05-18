"""
Extraction des résultats WandB pour les LLMs few-shot NLI.
Produit un CSV consolidé + des graphiques de comparaison.

Usage:
    python results/extract_wandb_results.py
"""

import os, re
import wandb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
ENTITY  = "reda300050-univer"
PROJECT = "fewshot-nli-fr"
OUT_DIR = os.path.join(os.path.dirname(__file__), "label_only_plots")
os.makedirs(OUT_DIR, exist_ok=True)

LLM_PATTERNS = {
    "Gemma 2 9B":       r"gemma",
    "Mistral 7B":       r"mistral_7b",
    "Llama 3 8B":       r"llama3",
    "Qwen 2.5 7B":      r"qwen25_7b",
    "DeepSeek R1 8B":   r"deepseek",
    "Phi-3 3.8B":       r"phi3",
    "Mistral NeMo 12B": r"mistral_nemo",
    "Mixtral 8x7B":     r"mixtral_8x7b",
    "Qwen 2.5 14B":     r"qwen25_14b",
}

# Nb de paramètres pour la courbe scaling
MODEL_PARAMS = {
    "Phi-3 3.8B":       3.8,
    "Qwen 2.5 7B":      7.0,
    "Mistral 7B":       7.0,
    "Llama 3 8B":       8.0,
    "DeepSeek R1 8B":   8.0,
    "Gemma 2 9B":       9.0,
    "Mistral NeMo 12B": 12.0,
    "Qwen 2.5 14B":     14.0,
    "Mixtral 8x7B":     47.0,
}

# ─────────────────────────────────────────────────────
# 1. EXTRACTION WANDB
# ─────────────────────────────────────────────────────
def extract_model(run_name: str) -> str:
    name_low = run_name.lower()
    for model, pattern in LLM_PATTERNS.items():
        if re.search(pattern, name_low):
            return model
    return None

def extract_metadata_from_name(run_name: str) -> dict:
    """Extrait train_ds, eval_ds, n_shots, seed depuis le nom du run."""
    # Format: model_train-DATASET_eval-DATASET_nSHOTS_sSEED
    meta = {}
    m = re.search(r"train-([a-z0-9\-]+)_eval-([a-z0-9\-]+)_n(\d+)_s(\d+)", run_name.lower())
    if m:
        meta["train_dataset"] = m.group(1)
        meta["eval_dataset"]  = m.group(2)
        meta["n_shots"]       = int(m.group(3))
        meta["seed"]          = int(m.group(4))
    return meta

def fetch_all_runs():
    api = wandb.Api(timeout=60)
    print(f"📡 Connexion à WandB ({ENTITY}/{PROJECT})...")

    records = []
    page = 0
    PAGE_SIZE = 100

    while True:
        try:
            runs = api.runs(
                f"{ENTITY}/{PROJECT}",
                order="-created_at",
                per_page=PAGE_SIZE,
                filters={"state": "finished"},
            )
            batch = []
            for r in runs:
                model = extract_model(r.name)
                if model is None:
                    continue
                acc = r.summary.get("test/accuracy")
                f1  = r.summary.get("test/f1_macro")
                pr  = r.summary.get("test/parse_rate")
                if acc is None:
                    continue
                meta = extract_metadata_from_name(r.name)
                if not meta:
                    continue
                batch.append({
                    "model":         model,
                    "n_params_B":    MODEL_PARAMS.get(model, 0),
                    "train_dataset": meta.get("train_dataset", "?"),
                    "eval_dataset":  meta.get("eval_dataset", "?"),
                    "n_shots":       meta.get("n_shots", -1),
                    "seed":          meta.get("seed", -1),
                    "accuracy":      round(acc, 4),
                    "f1_macro":      round(f1, 4) if f1 else None,
                    "parse_rate":    round(pr, 4) if pr else None,
                    "run_name":      r.name,
                })
            records.extend(batch)
            print(f"  Lot {page+1} : {len(batch)} runs LLM récupérés (total: {len(records)})")
            # L'itérateur wandb gère la pagination interne, on sort dès qu'on a tout
            break
        except Exception as e:
            print(f"  ⚠️  Erreur de connexion (lot {page+1}): {e}. Nouvelle tentative...")
            import time; time.sleep(5)
            page += 1
            if page > 5:
                print("  ❌ Trop d'erreurs, abandon.")
                break

    print(f"\n✅ {len(records)} runs extraits au total.")
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────
# 2. GRAPHIQUES
# ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10")
MODEL_ORDER = list(MODEL_PARAMS.keys())

def plot_accuracy_vs_shots(df: pd.DataFrame):
    """Courbe accuracy moyenne vs nombre de shots, par modèle."""
    avg = (
        df.groupby(["model", "n_shots"])["accuracy"]
        .mean()
        .reset_index()
    )
    present = [m for m in MODEL_ORDER if m in avg["model"].values]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in present:
        sub = avg[avg["model"] == model].sort_values("n_shots")
        ax.plot(sub["n_shots"], sub["accuracy"], marker="o", label=model, linewidth=2)
    
    ax.set_xlabel("Nombre de shots (Few-Shot)", fontsize=13)
    ax.set_ylabel("Accuracy moyenne", fontsize=13)
    ax.set_title("Performance Few-Shot par LLM (tous datasets confondus)", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticks([0, 1, 3, 5, 10])
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "accuracy_vs_shots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")

def plot_accuracy_vs_params(df: pd.DataFrame):
    """Scatter accuracy moyenne (5 shots) vs taille du modèle."""
    sub = df[df["n_shots"] == 5].groupby("model")["accuracy"].mean().reset_index()
    sub["n_params_B"] = sub["model"].map(MODEL_PARAMS)
    sub = sub.dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in sub.iterrows():
        ax.scatter(row["n_params_B"], row["accuracy"], s=200, zorder=5)
        ax.annotate(
            row["model"],
            (row["n_params_B"], row["accuracy"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9
        )
    
    ax.set_xlabel("Taille du modèle (Milliards de paramètres)", fontsize=13)
    ax.set_ylabel("Accuracy moyenne (5-shot)", fontsize=13)
    ax.set_title("Scaling : Performance vs Taille du Modèle (5-shot)", fontsize=14, fontweight="bold")
    ax.set_xlim(0)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "accuracy_vs_params.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")

def plot_cross_dataset_heatmap(df: pd.DataFrame):
    """Heatmap (train_ds x eval_ds) de l'accuracy moyenne pour chaque modèle."""
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        pivot = sub.groupby(["train_dataset", "eval_dataset"])["accuracy"].mean().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, vmin=0.3, vmax=1.0)
        ax.set_title(f"Heatmap Cross-Dataset — {model}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Dataset d'évaluation")
        ax.set_ylabel("Dataset d'entraînement")
        plt.tight_layout()
        fname = f"heatmap_{model.replace(' ', '_').lower()}.png"
        path = os.path.join(OUT_DIR, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  → {path}")

def plot_intra_vs_cross(df: pd.DataFrame):
    """Barplot comparatif intra-dataset vs cross-dataset par modèle."""
    df = df.copy()
    df["mode"] = df.apply(
        lambda r: "Intra-dataset" if r["train_dataset"] == r["eval_dataset"] else "Cross-dataset",
        axis=1
    )
    avg = df.groupby(["model", "mode"])["accuracy"].mean().reset_index()
    present = [m for m in MODEL_ORDER if m in avg["model"].values]
    avg = avg[avg["model"].isin(present)]
    avg["model"] = pd.Categorical(avg["model"], categories=present, ordered=True)
    avg = avg.sort_values("model")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=avg, x="model", y="accuracy", hue="mode", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy moyenne", fontsize=13)
    ax.set_title("Intra-dataset vs Cross-dataset par Modèle", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Mode")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "intra_vs_cross.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")

# ─────────────────────────────────────────────────────
# 3. TABLE DE SYNTHESE
# ─────────────────────────────────────────────────────
def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau de synthèse : accuracy et F1 moyens par modèle et nb de shots."""
    table = (
        df.groupby(["model", "n_shots"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std =("accuracy", "std"),
            f1_mean      =("f1_macro", "mean"),
            n_runs       =("accuracy", "count"),
        )
        .round(4)
        .reset_index()
    )
    return table

# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_all_runs()
    
    if df.empty:
        print("❌ Aucune donnée extraite. Vérifiez votre connexion WandB.")
        exit(1)
    
    # Sauvegarde CSV brut
    csv_raw = os.path.join(OUT_DIR, "llm_all_results.csv")
    df.to_csv(csv_raw, index=False)
    print(f"\n💾 CSV brut sauvegardé : {csv_raw} ({len(df)} lignes)")
    
    # Table de synthèse
    summary = build_summary_table(df)
    csv_summary = os.path.join(OUT_DIR, "llm_summary_table.csv")
    summary.to_csv(csv_summary, index=False)
    print(f"💾 Table de synthèse    : {csv_summary}")
    
    # Aperçu console
    print("\n=== SYNTHÈSE PAR MODÈLE (accuracy moyenne, tous shots/datasets) ===")
    global_avg = df.groupby("model")["accuracy"].mean().sort_values(ascending=False)
    print(global_avg.to_string())
    
    # Graphiques
    print("\n📊 Génération des graphiques...")
    plot_accuracy_vs_shots(df)
    plot_accuracy_vs_params(df)
    plot_intra_vs_cross(df)
    plot_cross_dataset_heatmap(df)
    
    print("\n✅ Tout est prêt dans results/label_only_plots/")
