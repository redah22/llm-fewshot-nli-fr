"""
Génère tous les dossiers de kernels Kaggle pour le projet NLI.

Structure générée :
  kaggle_kernels/
    {model}_{exp}/
      run.py
      kernel-metadata.json

Usage :
  python generate_kernels.py
  python generate_kernels.py --models camembert-xnli gpt2
  python generate_kernels.py --exps 1 4 7 8 9
"""

import os
import json
import argparse

KAGGLE_USERNAME = "colindvrt"
REPO_URL = "https://github.com/redah22/llm-fewshot-nli-fr.git"
REPO_DIR = "llm-fewshot-nli-fr"

MODELS = {
    "camembert-xnli": "camembert",
    "gpt2":           "gpt2",
    "flaubert-xnli":  "flaubert",
    # "flaubert-custom": "flaubert_custom",  # décommenter quand dispo sur HF
}

EXPERIMENTS = list(range(1, 12))  # EXP 1 à 11

EXP_DESCRIPTIONS = {
    1:  "FraCaS(GQ) train | val+test=GQNLI",
    2:  "FraCaS(GQ 85/15) train/val | test=GQNLI",
    3:  "FraCaS(GQ)+SICK_train | val=FraCaS+SICK_dev | test=GQNLI+SICK",
    4:  "GQNLI(80%) train | val=GQNLI(20%) | test=FraCaS",
    5:  "GQNLI+SICK train | val=GQNLI+SICK_dev | test=FraCaS+SICK",
    6:  "FraCaS+GQNLI(20%) train | val mix | test=GQNLI(70%)",
    7:  "FraCaS complet | val=SICK_dev | test=SICK_test",
    8:  "N-shot SICK (10/25/50/100/200 shots)",
    9:  "RTE3 intra (80/20 premise) | test=RTE3-test",
    10: "RTE3+DACCORD binary | test=RTE3-test",
    11: "RTE3 binary | test=DACCORD+RTE3-test",
}


def make_run_py(model_arg: str, exp: int) -> str:
    n_runs = 10 if exp == 8 else 8
    return f'''\
"""
Kernel Kaggle — {model_arg.upper()} EXP{exp}
{EXP_DESCRIPTIONS[exp]} | {n_runs} runs sweep LoRA
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
    print(f"Impossible de charger WANDB_API_KEY depuis les secrets : {{e}}")
    print("Assurez-vous d\'avoir ajouté WANDB_API_KEY dans Add-ons > Secrets sur Kaggle.")
    sys.exit(1)

# ── 3. Clone du repo ────────────────────────────────────────
REPO_URL = "{REPO_URL}"
REPO_DIR = "{REPO_DIR}"

if not os.path.exists(REPO_DIR):
    print(f"Clonage de {{REPO_URL}}...")
    subprocess.run(["git", "clone", REPO_URL], check=True)
else:
    print("Repo déjà présent, mise à jour...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

os.chdir(REPO_DIR)
print(f"Répertoire courant : {{os.getcwd()}}")

# ── 4. Lancement du sweep ────────────────────────────────────
print("\\n" + "="*60)
print("LANCEMENT : {model_arg.upper()} | EXP {exp} | {n_runs} runs")
print("="*60 + "\\n")

subprocess.run([
    sys.executable,
    "experiments/lora/sweep_classifiers.py",
    "{model_arg}",
    "{exp}",
    "auto"
], check=True)

print("\\nTerminé !")
'''


def make_metadata(model_arg: str, model_short: str, exp: int) -> dict:
    # Le slug doit correspondre exactement au titre slugifié par Kaggle
    slug = f"nli-{model_arg}-exp{exp}"   # ex: nli-camembert-xnli-exp1
    title = f"NLI {model_arg.upper()} EXP{exp}"
    return {
        "id": f"{KAGGLE_USERNAME}/{slug}",
        "title": title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "accelerator": "nvidiaTeslaT4x2",
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def generate(models=None, exps=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = models or list(MODELS.keys())
    exps = exps or EXPERIMENTS

    created = []
    for model_arg in models:
        if model_arg not in MODELS:
            print(f"[SKIP] Modèle inconnu : {model_arg}")
            continue
        model_short = MODELS[model_arg]
        for exp in exps:
            if exp not in EXP_DESCRIPTIONS:
                print(f"[SKIP] EXP {exp} inconnue")
                continue

            dir_name = f"{model_short}_exp{exp}"
            dir_path = os.path.join(base_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)

            run_path = os.path.join(dir_path, "run.py")
            meta_path = os.path.join(dir_path, "kernel-metadata.json")

            # Ne pas écraser si déjà à jour
            run_content = make_run_py(model_arg, exp)
            with open(run_path, "w") as f:
                f.write(run_content)

            meta = make_metadata(model_arg, model_short, exp)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            created.append(dir_name)
            print(f"  [OK] {dir_name}/")

    print(f"\n{len(created)} kernels générés.")
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère les kernels Kaggle")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Modèles à générer (ex: camembert-xnli gpt2 flaubert-xnli)")
    parser.add_argument("--exps", nargs="+", type=int, default=EXPERIMENTS,
                        help="Expériences à générer (ex: 1 4 7 8 9)")
    args = parser.parse_args()

    print(f"Génération des kernels pour : {args.models}")
    print(f"Expériences : {args.exps}\n")
    generate(args.models, args.exps)
