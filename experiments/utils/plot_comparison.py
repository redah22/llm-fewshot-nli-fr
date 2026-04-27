"""
Graphique de comparaison des méthodes PEFT
==========================================

Génère un graphique à barres comparant les performances de :
  - Full Fine-tuning (CamemBERT, FlauBERT)
  - LoRA (CamemBERT, FlauBERT)
  - IA³  (CamemBERT, FlauBERT)

Lit automatiquement tous les fichiers JSON dans results/metrics/
et les résultats fixes connus (full fine-tuning).

Utilisation :
    python3 experiments/utils/plot_comparison.py
    python3 experiments/utils/plot_comparison.py --exp 1   (filtre EXP 1 uniquement)

Sortie :
    results/metricsgraphs/comparison_all.png
    results/metricsgraphs/comparison_exp1.png  (si --exp 1)
"""

import os
import sys
import json
import glob
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────
# 0. RÉSULTATS FIXES — Full Fine-tuning
#    (issus des expériences déjà réalisées, mis à jour manuellement)
# ─────────────────────────────────────────────────────────

FULL_FINETUNING_RESULTS = {
    # Format : (modèle, exp_label, train_dataset, test_dataset, accuracy)
    # EXP 1 — FraCaS → GQNLI
    ("CamemBERT", "exp1", "FraCaS→GQNLI"): 0.373,
    ("FlauBERT",  "exp1", "FraCaS→GQNLI"): 0.333,
    # EXP 2 — GQNLI → FraCaS
    ("CamemBERT", "exp2", "GQNLI→FraCaS"): None,  # À remplir après expérience
    ("FlauBERT",  "exp2", "GQNLI→FraCaS"): None,
    # EXP 3 — RTE3 → DACCORD
    ("CamemBERT", "exp3", "RTE3→DACCORD"): None,
    ("FlauBERT",  "exp3", "RTE3→DACCORD"): None,
}

# ─────────────────────────────────────────────────────────
# 1. ARGUMENTS
# ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Graphique de comparaison PEFT vs Full")
parser.add_argument("--exp", type=str, default=None,
                    help="Filtrer par expérience (ex: '1', '2', '3'). Défaut = tous.")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────
# 2. LECTURE DES FICHIERS JSON DE RÉSULTATS METRICS
# ─────────────────────────────────────────────────────────

metrics_dir = "results/metrics"
if not os.path.isdir(metrics_dir):
    print(f"❌ Dossier {metrics_dir} introuvable. Lancez d'abord des expériences.")
    exit(1)

json_files = sorted(glob.glob(os.path.join(metrics_dir, "*.json")))
if not json_files:
    print(f"⚠️  Aucun fichier JSON trouvé dans {metrics_dir}.")
    print("   Lancez d'abord : python3 experiments/lora/cross_dataset_ia3.py (ou cross_dataset_lora.py)")

# Structure de collecte :
# data[exp_id][scenario] = [(method_label, color, accuracy), ...]
# ex: data["exp1"]["FraCaS→GQNLI"] = [("CamemBERT LoRA", "#3498db", 0.42), ...]

data_by_exp = {}  # exp_id -> scenario -> list of (label, color, acc)

EXP_LABELS = {
    "exp1": "EXP 1 : FraCaS → GQNLI",
    "exp2": "EXP 2 : GQNLI → FraCaS",
    "exp3_daccord":  "EXP 3 : RTE3 → DACCORD",
    "exp3_rte3test": "EXP 3 : RTE3 → RTE3-TEST",
}

METHOD_COLORS = {
    "Full Fine-tuning": "#2c3e50",
    "LoRA":             "#3498db",
    "IA³":              "#e74c3c",
    "Baseline":         "#95a5a6",
    "XNLI Transfer":    "#9b59b6",
}

def parse_model(model_str):
    """Renvoie 'CamemBERT' ou 'FlauBERT' depuis le chemin du modèle."""
    m = model_str.lower()
    if "camembert" in m:
        return "CamemBERT"
    if "flaubert" in m:
        return "FlauBERT"
    return model_str.split("/")[-1]

def get_peft_method(result):
    """Détecte la méthode PEFT depuis le JSON."""
    if "peft_method" in result:
        return result["peft_method"]
    exp = result.get("experiment", "")
    if "lora" in exp.lower():
        return "LoRA"
    if "ia3" in exp.lower():
        return "IA³"
    return "Full Fine-tuning"


for jf in json_files:
    try:
        with open(jf) as f:
            r = json.load(f)
    except Exception:
        continue

    exp_raw = r.get("experiment", "")
    method  = get_peft_method(r)
    model   = parse_model(r.get("model", ""))

    # ── EXP 1 ──
    if "exp1" in exp_raw and r.get("final_accuracy") is not None:
        scenario = "FraCaS→GQNLI"
        key = "exp1"
        data_by_exp.setdefault(key, {}).setdefault(scenario, [])
        label = f"{model} {method}"
        data_by_exp[key][scenario].append((label, method, model, r["final_accuracy"]))

    # ── EXP 2 ──
    elif "exp2" in exp_raw and r.get("final_accuracy") is not None:
        scenario = "GQNLI→FraCaS"
        key = "exp2"
        data_by_exp.setdefault(key, {}).setdefault(scenario, [])
        label = f"{model} {method}"
        data_by_exp[key][scenario].append((label, method, model, r["final_accuracy"]))

    # ── EXP 3 ──
    elif "exp3" in exp_raw:
        dacc = r.get("daccord", {}).get("accuracy")
        rte3 = r.get("rte3_test", {}).get("accuracy")
        label = f"{model} {method}"
        if dacc is not None:
            key = "exp3"
            data_by_exp.setdefault(key, {}).setdefault("RTE3→DACCORD", [])
            data_by_exp[key]["RTE3→DACCORD"].append((label, method, model, dacc))
        if rte3 is not None:
            key = "exp3"
            data_by_exp.setdefault(key, {}).setdefault("RTE3→RTE3-TEST", [])
            data_by_exp[key]["RTE3→RTE3-TEST"].append((label, method, model, rte3))

# Injecter les résultats Full Fine-tuning connus
for (model_name, exp_id, scenario), acc in FULL_FINETUNING_RESULTS.items():
    if acc is None:
        continue
    data_by_exp.setdefault(exp_id, {}).setdefault(scenario, [])
    label = f"{model_name} Full FT"
    # Éviter les doublons
    existing_labels = [x[0] for x in data_by_exp[exp_id][scenario]]
    if label not in existing_labels:
        data_by_exp[exp_id][scenario].append((label, "Full Fine-tuning", model_name, acc))


# ─────────────────────────────────────────────────────────
# 3. FILTRAGE PAR EXPÉRIENCE
# ─────────────────────────────────────────────────────────

if args.exp:
    filter_key = f"exp{args.exp}"
    data_by_exp = {k: v for k, v in data_by_exp.items() if k == filter_key}
    if not data_by_exp:
        print(f"⚠️  Aucune donnée trouvée pour l'expérience {args.exp}.")
        exit(0)

if not data_by_exp:
    print("⚠️  Aucune donnée exploitable trouvée. Vérifiez results/metrics/ et les résultats Full FT.")
    print("   ℹ️  Une figure vide 'aucune donnée' sera générée.")

# ─────────────────────────────────────────────────────────
# 4. CRÉATION DES GRAPHIQUES
# ─────────────────────────────────────────────────────────

# Calculer le nombre total de sous-graphiques nécessaires
all_scenarios = [(exp, scen, entries)
                 for exp, scenarios in sorted(data_by_exp.items())
                 for scen, entries in sorted(scenarios.items())]

n_plots = max(len(all_scenarios), 1)
ncols = min(n_plots, 3)
nrows = (n_plots + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
if n_plots == 1:
    axes = np.array([[axes]])
elif nrows == 1:
    axes = axes.reshape(1, -1)


HATCHES = {
    "CamemBERT": "",
    "FlauBERT":  "///",
}

def get_color_for_method(method_str):
    for key in METHOD_COLORS:
        if key.lower() in method_str.lower():
            return METHOD_COLORS[key]
    return "#7f8c8d"


for idx, (exp_id, scenario, entries) in enumerate(all_scenarios):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]

    # Trier pour un affichage cohérent : Full FT d'abord, puis LoRA, puis IA³
    METHOD_ORDER = {"Full Fine-tuning": 0, "Full FT": 0, "LoRA": 1, "IA³": 2, "IA3": 2, "Baseline": 3}
    entries_sorted = sorted(entries, key=lambda x: (METHOD_ORDER.get(x[1], 9), x[2]))

    labels  = [e[0] for e in entries_sorted]
    methods = [e[1] for e in entries_sorted]
    models  = [e[2] for e in entries_sorted]
    accs    = [e[3] * 100 for e in entries_sorted]  # En %
    colors  = [get_color_for_method(m) for m in methods]
    hatches = [HATCHES.get(m, "") for m in models]

    x = np.arange(len(labels))
    bars = ax.bar(x, accs, color=colors, hatch=hatches, edgecolor="white",
                  linewidth=1.5, width=0.6, alpha=0.88)

    # Annotations des valeurs sur chaque barre
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc:.1f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    # Ligne de baseline (hasard = 33%)
    ax.axhline(y=33.3, color="#e74c3c", linestyle=":", linewidth=1.2, alpha=0.6,
               label="Hasard (33%)")

    ax.set_title(scenario, fontweight="bold", fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Masquer les axes vides
for idx in range(len(all_scenarios), nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_visible(False)

# Légende globale
legend_patches = [
    mpatches.Patch(color=c, label=m) for m, c in METHOD_COLORS.items()
]
hatch_patches = [
    mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="FlauBERT"),
    mpatches.Patch(facecolor="white", edgecolor="black", label="CamemBERT"),
]
fig.legend(
    handles=legend_patches + hatch_patches,
    loc="lower center",
    ncol=min(len(legend_patches) + 2, 4),
    fontsize=9,
    bbox_to_anchor=(0.5, -0.04),
    title="Méthode  |  Hachures = FlauBERT",
    title_fontsize=9,
)

exp_suffix = f"_exp{args.exp}" if args.exp else "_all"
fig.suptitle(
    "Comparaison des méthodes de fine-tuning NLI (précision sur le jeu de test cross-dataset)",
    fontsize=13, fontweight="bold", y=1.01
)

plt.tight_layout()
os.makedirs("results/metricsgraphs", exist_ok=True)
output_path = f"results/metricsgraphs/comparison{exp_suffix}.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✅ Graphique de comparaison sauvegardé : {output_path}")
print("\n📊 Données utilisées :")
for exp_id, scenarios in sorted(data_by_exp.items()):
    for scen, entries in sorted(scenarios.items()):
        print(f"  [{exp_id}] {scen}")
        for label, method, model, acc in entries:
            print(f"     • {label:<30} : {acc:.1%}")
