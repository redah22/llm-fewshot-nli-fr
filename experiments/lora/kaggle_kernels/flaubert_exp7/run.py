"""
Kernel Kaggle — FLAUBERT-XNLI EXP7
FraCaS complet | val=SICK_dev | test=SICK_test | 8 runs sweep LoRA
"""

import subprocess
import sys
import os

# ── 1. Dépendances ──────────────────────────────────────────
print("Installation des dépendances...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "wandb", "transformers>=4.40", "peft>=0.10",
    "datasets", "scikit-learn", "accelerate",
    "bitsandbytes>=0.43"
], check=True)

# ── 2. WandB API Key (Kaggle Secret) ────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    wandb_key = UserSecretsClient().get_secret("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = wandb_key
    print("WandB API key chargée depuis Kaggle Secrets.")
except Exception as e:
    print(f"Impossible de charger WANDB_API_KEY depuis les secrets : {e}")
    print("Assurez-vous d'avoir ajouté WANDB_API_KEY dans Add-ons > Secrets sur Kaggle.")
    sys.exit(1)

# ── 3. Clone du repo ────────────────────────────────────────
REPO_URL = "https://github.com/redah22/llm-fewshot-nli-fr.git"
REPO_DIR = "llm-fewshot-nli-fr"

if not os.path.exists(REPO_DIR):
    print(f"Clonage de {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL], check=True)
else:
    print("Repo déjà présent, mise à jour...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

os.chdir(REPO_DIR)
print(f"Répertoire courant : {os.getcwd()}")

# ── 4. Lancement du sweep ────────────────────────────────────
print("\n" + "="*60)
print("LANCEMENT : FLAUBERT-XNLI | EXP 7 | 8 runs")
print("="*60 + "\n")

subprocess.run([
    sys.executable,
    "experiments/lora/sweep_classifiers.py",
    "flaubert-xnli",
    "7",
    "auto"
], check=True)

print("\nTerminé !")
