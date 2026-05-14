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
    
    # Récupérer tous les runs du projet
    # On filtre par tag ou par config si possible, sinon on filtre après
    runs = api.runs(f"{PROJECT_NAME}")
    
    data = []
    for run in runs:
        # On ne prend que les runs "Label-Only" (pas de CoT)
        # Et on vérifie que le run est terminé
        if run.state != "finished":
            continue
            
        cfg = run.config
        # Vérifier si c'est un run Label-Only (selon les nouveaux scripts)
        if cfg.get("use_cot") is False:
            summary = run.summary
            
            data.append({
                "model": cfg.get("model_short"),
                "n_shots": cfg.get("n_shots"),
                "train_ds": cfg.get("train_dataset"),
                "eval_ds": cfg.get("eval_dataset"),
                "mode": cfg.get("mode", "intra"),
                "seed": cfg.get("seed"),
                "accuracy": summary.get("test/accuracy"),
                "f1_macro": summary.get("test/f1_macro"),
                "parse_rate": summary.get("test/parse_rate", 1.0)
            })
            
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
