"""
Génération des graphiques d'entraînement
=========================================

Lit le fichier JSON de résultats produit par eval_daccord_binary.py
et génère un fichier PNG avec 4 graphiques :

  1. Train Loss vs Eval Loss par epoch  → détecte le sur-apprentissage
  2. Eval Accuracy par epoch            → voit quand le modèle plateau
  3. Learning Rate par step             → vérifie la descente vers 0
  4. Grad Norm par step                 → détecte les instabilités

Utilisation :
    python3 experiments/utils/plot_training.py
    python3 experiments/utils/plot_training.py results/mon_fichier.json

Le graphique est sauvegardé dans : results/<nom_du_fichier>.png
"""

import json
import sys
import os
import matplotlib
matplotlib.use("Agg")  # Mode sans interface graphique (pour serveurs)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────
# 1. CHARGEMENT DU FICHIER JSON
# ─────────────────────────────────────────────────────────

# Chercher le fichier JSON à utiliser
if len(sys.argv) > 1:
    json_path = sys.argv[1]
else:
    # Chercher automatiquement le dernier fichier JSON dans results/
    result_files = [
        f for f in os.listdir("results/metrics")
        if f.endswith(".json") and "training_history" in
        json.load(open(f"results/metrics/{f}")).keys()
    ]
    if not result_files:
        print("❌ Aucun fichier de résultats avec historique d'entraînement trouvé.")
        print("   Lancez d'abord : python3 experiments/fine_tuning/eval_daccord_binary.py")
        exit(1)
    # Prendre le plus récent
    json_path = "results/metrics/" + sorted(
        result_files,
        key=lambda f: os.path.getmtime(f"results/metrics/{f}"),
        reverse=True
    )[0]

print(f"📂 Chargement : {json_path}")
with open(json_path, "r", encoding="utf-8") as f:
    results = json.load(f)

# Extraire les données
model_name    = results.get("model", "modèle")
dataset_name  = results.get("dataset", "dataset")
baseline_acc  = results.get("baseline_accuracy", None)
final_acc     = results.get("final_accuracy", None)
best_epoch    = results.get("best_epoch", None)
best_val_acc  = results.get("best_val_accuracy", None)
epochs_trained = results.get("epochs_trained", None)

history       = results.get("training_history", {})
train_steps   = history.get("train_steps", [])
eval_epochs   = history.get("eval_epochs", [])

if not train_steps and not eval_epochs:
    print("❌ Pas d'historique d'entraînement dans ce fichier.")
    exit(1)

# ─────────────────────────────────────────────────────────
# 2. PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────────────────

# Train (par step)
train_epochs_x = [s["epoch"] for s in train_steps]
train_loss_y   = [s["loss"] for s in train_steps]
grad_norm_y    = [s["grad_norm"] for s in train_steps if s.get("grad_norm") is not None]
grad_norm_x    = [s["epoch"] for s in train_steps if s.get("grad_norm") is not None]
lr_y           = [s["learning_rate"] for s in train_steps if s.get("learning_rate") is not None]
lr_x           = [s["epoch"] for s in train_steps if s.get("learning_rate") is not None]

# Eval (par epoch)
# Filtrer les évaluations doublons (le test final a souvent le même numéro d'epoch que la dernière validation)
unique_evals = []
seen_epochs = set()
for e in eval_epochs:
    if e["epoch"] not in seen_epochs:
        seen_epochs.add(e["epoch"])
        unique_evals.append(e)

eval_epoch_x   = [e["epoch"] for e in unique_evals]
eval_loss_y    = [e["eval_loss"] for e in unique_evals]
eval_acc_y     = [e["eval_accuracy"] * 100 for e in unique_evals]  # En %

print(f"   Steps d'entraînement collectés : {len(train_steps)}")
print(f"   Epochs d'évaluation collectées : {len(eval_epochs)}")

# ─────────────────────────────────────────────────────────
# 3. CRÉATION DE LA FIGURE
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Courbes d'entraînement — {model_name.split('/')[-1]} sur {dataset_name}\n"
    f"Epochs entraînées : {epochs_trained}  |  "
    f"Meilleure epoch : {best_epoch}  |  "
    f"Meilleure val acc : {f'{best_val_acc:.1%}' if best_val_acc else 'N/A'}",
    fontsize=13, fontweight="bold", y=1.02
)

COLORS = {
    "train":    "#3498db",   # Bleu
    "eval":     "#e74c3c",   # Rouge
    "lr":       "#2ecc71",   # Vert
    "grad":     "#f39c12",   # Orange
    "baseline": "#95a5a6",   # Gris
    "best":     "#9b59b6",   # Violet
}

# ── Graphique 1 : Train Loss vs Eval Loss ──────────────────
ax1 = axes[0, 0]
if train_loss_y:
    ax1.plot(train_epochs_x, train_loss_y,
             color=COLORS["train"], alpha=0.7, linewidth=1.5,
             label="Train Loss (par step)")
if eval_loss_y:
    ax1.plot(eval_epoch_x, eval_loss_y,
             color=COLORS["eval"], linewidth=2, marker="o", markersize=5,
             label="Eval Loss (par epoch)")
if best_epoch:
    ax1.axvline(x=best_epoch, color=COLORS["best"], linestyle="--",
                alpha=0.7, label=f"Meilleure epoch ({best_epoch})")

ax1.set_title("Loss : Train vs Validation", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Note sur-apprentissage
if len(eval_loss_y) > 3:
    min_idx = np.argmin(eval_loss_y)
    if min_idx < len(eval_loss_y) - 1:
        ax1.annotate(
            "← sur-apprentissage possible",
            xy=(eval_epoch_x[min_idx], eval_loss_y[min_idx]),
            xytext=(eval_epoch_x[min_idx] + 0.5, eval_loss_y[min_idx] + 0.05),
            fontsize=8, color=COLORS["best"],
            arrowprops=dict(arrowstyle="->", color=COLORS["best"]),
        )

# ── Graphique 2 : Eval Accuracy par epoch ──────────────────
ax2 = axes[0, 1]
if eval_acc_y:
    ax2.plot(eval_epoch_x, eval_acc_y,
             color=COLORS["eval"], linewidth=2, marker="o", markersize=6,
             label="Validation Accuracy")
    # Marquer le meilleur point
    best_idx = np.argmax(eval_acc_y)
    ax2.scatter([eval_epoch_x[best_idx]], [eval_acc_y[best_idx]],
                color=COLORS["best"], zorder=5, s=100,
                label=f"Meilleure : {eval_acc_y[best_idx]:.1f}% (epoch {eval_epoch_x[best_idx]})")

# Ligne de baseline
if baseline_acc is not None:
    ax2.axhline(y=baseline_acc * 100, color=COLORS["baseline"],
                linestyle="--", alpha=0.8,
                label=f"Baseline : {baseline_acc:.1%}")

# NOTE : la ligne du test final est volontairement omise
# (elle apparaît après le training et crée un pic trompeur sur le graphique)

ax2.set_title("Accuracy sur la Validation par Epoch", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Graphique 3 : Learning Rate ────────────────────────────
ax3 = axes[1, 0]
if lr_y:
    ax3.plot(lr_x, lr_y,
             color=COLORS["lr"], linewidth=1.5, alpha=0.9,
             label="Learning Rate")
    ax3.fill_between(lr_x, lr_y, alpha=0.15, color=COLORS["lr"])

ax3.set_title("Learning Rate (décroissance vers 0)", fontweight="bold")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))

# ── Graphique 4 : Grad Norm ────────────────────────────────
ax4 = axes[1, 1]
if grad_norm_y:
    ax4.plot(grad_norm_x, grad_norm_y,
             color=COLORS["grad"], linewidth=1, alpha=0.8,
             label="Grad Norm (par step)")
    # Ligne de moyenne mobile (fenêtre 5)
    if len(grad_norm_y) >= 5:
        window = 5
        smooth = np.convolve(grad_norm_y, np.ones(window)/window, mode="valid")
        smooth_x = grad_norm_x[window-1:]
        ax4.plot(smooth_x, smooth,
                 color="darkred", linewidth=2,
                 label=f"Moyenne mobile (fenêtre {window})")

ax4.set_title("Grad Norm — Stabilité de l'entraînement", fontweight="bold")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Grad Norm")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Note si spike détecté
if grad_norm_y:
    max_norm = max(grad_norm_y)
    mean_norm = np.mean(grad_norm_y)
    if max_norm > 5 * mean_norm:
        ax4.annotate(
            f"⚠ Spike : {max_norm:.1f}",
            xy=(grad_norm_x[np.argmax(grad_norm_y)], max_norm),
            xytext=(grad_norm_x[np.argmax(grad_norm_y)] + 0.5, max_norm * 0.9),
            fontsize=8, color="red",
        )

# ─────────────────────────────────────────────────────────
# 4. SAUVEGARDE
# ─────────────────────────────────────────────────────────

plt.tight_layout()

os.makedirs("results/metricsgraphs", exist_ok=True)
base_name = os.path.splitext(os.path.basename(json_path))[0]
output_path = f"results/metricsgraphs/{base_name}_training_curves.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✅ Graphique sauvegardé : {output_path}")
print(f"\n📊 Résumé de l'entraînement :")
print(f"   Baseline (test)         : {baseline_acc:.1%}" if baseline_acc else "")
print(f"   Meilleure val accuracy  : {best_val_acc:.1%} (epoch {best_epoch})" if best_val_acc else "")
print(f"   Résultat final (test)   : {final_acc:.1%}" if final_acc else "")
print(f"   Gain total              : {final_acc - baseline_acc:+.1%}" if (final_acc and baseline_acc) else "")
