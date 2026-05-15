"""
Exportation et Visualisation Automatisée des résultats Label-Only Few-Shot.
Ce script se connecte à WandB, télécharge les résultats des runs LLM
et génère des graphiques comparatifs pertinents pour le TER.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from dotenv import load_dotenv

# Charger les variables d'environnement (WANDB_API_KEY)
load_dotenv()

# Configuration
PROJECT_NAME = "fewshot-nli-fr"
ENTITY = None # Remplacer par votre username wandb si nécessaire
OUTPUT_DIR = "results/label_only_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_wandb_data():
    print(f"🔄 Connexion à WandB (Projet: {PROJECT_NAME})...")
    api = wandb.Api()
    
    # On ne garde que les runs "finished" pour gagner du temps
    filters = {"state": "finished"}
    runs = api.runs(f"reda300050-univer/{PROJECT_NAME}", filters=filters)
    
    data = []
    print("📥 Récupération des runs (cela peut prendre un peu de temps) : ", end="", flush=True)
    for i, run in enumerate(runs):
        if i % 10 == 0:
            print(".", end="", flush=True)
            
        cfg = run.config
        summary = run.summary
        
        # 1. Runs Label-Only (LLM) - par config OU par nom
        if cfg.get("use_cot") is False or cfg.get("mode") in ["intra", "cross"] or "hf_name" in str(cfg.get("model", "")) or any(m in run.name for m in ["deepseek", "llama", "qwen", "gemma", "mistral"]):
            
            # Essayer de lire depuis la config
            model_name = cfg.get("model", "Unknown")
            train_ds = cfg.get("train_dataset")
            eval_ds = cfg.get("eval_dataset")
            mode = cfg.get("mode")
            n_shots = cfg.get("n_shots")
            seed = cfg.get("seed")
            
            # Si la config est vide, parser le nom du run (ex: deepseek_r1_8b_train-rte3_eval-fracas-gq_n1_s999)
            if not train_ds and "_train-" in run.name:
                try:
                    parts = run.name.split("_")
                    model_name = parts[0] + "_" + parts[1] if len(parts) > 1 else parts[0]
                    
                    for p in parts:
                        if p.startswith("train-"): train_ds = p.replace("train-", "")
                        elif p.startswith("eval-"): eval_ds = p.replace("eval-", "")
                        elif p.startswith("n") and p[1:].isdigit(): n_shots = int(p[1:])
                        elif p.startswith("s") and p[1:].isdigit(): seed = int(p[1:])
                except:
                    pass
            
            if not mode:
                mode = "intra" if train_ds == eval_ds else "cross"

            
        # 2. Runs PEFT (IA3 / LoRA)
        elif cfg.get("peft_method") or summary.get("peft_method"):
            peft = cfg.get("peft_method") or summary.get("peft_method")
            model_name = f"{cfg.get('model_key', summary.get('model', 'model'))} ({peft})"
            train_ds = cfg.get("train_ds") or summary.get("train_ds")
            eval_ds = cfg.get("test_ds") or summary.get("test_ds") or train_ds
            mode = cfg.get("mode") or summary.get("mode") or ("intra" if train_ds == eval_ds else "cross")
            n_shots = cfg.get("n_shots") or summary.get("n_shots")
            seed = cfg.get("seed_data") or summary.get("seed_data") or 42
            
        else:
            print(f"Skipping {run.name}: Unrecognized run type (no peft_method or LLM signature)")
            continue
            
        # Sécurité pour certains vieux runs
        acc = summary.get("test/accuracy") or summary.get("eval/accuracy") or summary.get("final_test_accuracy")
        f1 = summary.get("test/f1_macro") or summary.get("eval/f1_score") or summary.get("test/f1_score") or summary.get("final_test_f1_score")
        
        if acc is None or f1 is None:
            print(f"Skipping {run.name}: Missing metrics (acc={acc}, f1={f1})")
            continue
            
        data.append({
            "model": model_name,
            "n_shots": n_shots or 0,
            "train_ds": train_ds or "unknown",
            "eval_ds": eval_ds or "unknown",
            "mode": mode or "intra",
            "seed": seed or 42,
            "accuracy": acc,
            "f1_macro": f1,
            "parse_rate": summary.get("test/parse_rate", 1.0)
        })

    print(f"\n✅ {len(data)} runs valides récupérés.")
    df = pd.DataFrame(data)
    if df.empty:
        print("⚠️ Aucun résultat Label-Only trouvé sur WandB.")
        return None
        
    print(f"✅ {len(df)} runs récupérés.")
    return df

def plot_intra_dataset_comparison(df):
    """Génère une courbe comparant les LLMs en mode Intra-Dataset."""
    intra_df = df[df["mode"] == "intra"].copy()
    if intra_df.empty: return

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Palette de couleurs distinctes
    palette = sns.color_palette("husl", n_colors=len(intra_df["model"].unique()))
    
    sns.lineplot(
        data=intra_df, 
        x="n_shots", y="accuracy", hue="model", 
        marker="o", err_style="bars", palette=palette, linewidth=2
    )
    
    plt.axhline(y=1/3, color='r', linestyle='--', alpha=0.5, label="Hasard (33%)")
    plt.title("Performance Intra-Dataset (Label-Only Few-Shot)\nAccuracy vs Nombre d'exemples", fontsize=15)
    plt.xlabel("Nombre d'exemples (n_shots)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title="Modèle", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, "intra_dataset_comparison.png")
    plt.savefig(path, dpi=300)
    print(f"📊 Graphique sauvegardé : {path}")

def plot_cross_dataset_heatmap(df):
    """Génère une heatmap des performances Cross-Dataset pour un n_shots donné."""
    # On prend le n_shots le plus élevé pour voir le transfert max
    max_shots = df["n_shots"].max()
    cross_df = df[df["n_shots"] == max_shots].copy()
    
    if cross_df.empty: return

    # Pour chaque modèle, on peut faire une heatmap
    for model in cross_df["model"].unique():
        model_df = cross_df[cross_df["model"] == model]
        
        # Pivoter les données pour la heatmap
        pivot = model_df.pivot_table(
            index="train_ds", columns="eval_ds", values="accuracy", aggfunc="mean"
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0, vmax=1)
        plt.title(f"Matrice de Transfert Cross-Dataset ({model})\nAccuracy à {max_shots} shots", fontsize=14)
        plt.xlabel("Dataset d'Évaluation", fontsize=12)
        plt.ylabel("Dataset d'Entraînement (Few-shot)", fontsize=12)
        plt.tight_layout()
        
        path = os.path.join(OUTPUT_DIR, f"heatmap_cross_{model}.png")
        plt.savefig(path, dpi=300)
        print(f"📊 Heatmap sauvegardée : {path}")

def main():
    print("🚀 Démarrage de l'exportation automatique...")
    
    df = fetch_wandb_data()
    if df is not None:
        # Sauvegarder les données brutes
        df.to_csv(os.path.join(OUTPUT_DIR, "label_only_results.csv"), index=False)
        print(f"💾 Données brutes sauvegardées dans {OUTPUT_DIR}/label_only_results.csv")
        
        # Générer les courbes
        plot_intra_dataset_comparison(df)
        plot_cross_dataset_heatmap(df)
        
        print("\n✨ Automatisation terminée ! Tous les graphiques sont dans results/label_only_plots/")

if __name__ == "__main__":
    main()
