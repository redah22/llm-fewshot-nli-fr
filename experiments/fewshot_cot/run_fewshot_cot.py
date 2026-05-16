"""
Few-Shot + Chain-of-Thought NLI — Modèles génératifs (HuggingFace, OpenAI, Google)

Pas de fine-tuning : inférence pure avec N exemples dans le prompt.

Modèles supportés :
  llama3      → meta-llama/Llama-3.1-8B-Instruct         (HuggingFace, 4-bit)
  qwen2.5     → Qwen/Qwen2.5-7B-Instruct                 (HuggingFace, 4-bit)
  deepseek-r1 → deepseek-ai/DeepSeek-R1-Distill-Llama-8B (HuggingFace, 4-bit)
  mistral     → mistralai/Mistral-7B-Instruct-v0.3        (HuggingFace, 4-bit)
  magistral   → mistralai/Magistral-Small-24B-Instruct-2506 (HuggingFace, 4-bit)
  gpt-4o-mini → gpt-4o-mini                              (OpenAI API — OPENAI_API_KEY)
  gpt-4o      → gpt-4o                                   (OpenAI API — OPENAI_API_KEY)
  gemma       → google/gemma-4-E4B-it                    (HuggingFace, 4-bit)
  lucie       → OpenLLM-France/Lucie-7B-Instruct-v1.1    (HuggingFace, 4-bit — français natif)
  aya         → CohereForAI/aya-expanse-8b               (HuggingFace, 4-bit — multilingue)

Variables d'environnement :
  OPENAI_API_KEY   pour les modèles GPT

Datasets : fracas-gq | gqnli | sick | rte3 | daccord

Usage :
  python3 run_fewshot_cot.py --model qwen2.5 --dataset gqnli --n_shots 5
  python3 run_fewshot_cot.py --model llama3 --dataset sick --n_shots 3 --no_cot
  python3 run_fewshot_cot.py --model gpt-4o-mini --dataset gqnli --n_shots 5
  python3 run_fewshot_cot.py --model gemma --dataset rte3 --n_shots 3
"""

import os, sys, random, argparse, json, re
from collections import Counter, defaultdict

os.environ["WANDB_PROJECT"] = "fewshot-nli-fr"

import numpy as np
import torch
import wandb
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

# ─────────────────────────────────────────────────────────
# 1. CONFIGURATIONS MODÈLES
# ─────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    # ── HuggingFace (local, 4-bit) ────────────────────────
    "llama3": {
        "hf_name":  "meta-llama/Llama-3.1-8B-Instruct",
        "short":    "llama3_8b",
        "use_4bit": True,
        "backend":  "hf",
    },
    "qwen2.5": {
        "hf_name":  "Qwen/Qwen2.5-7B-Instruct",
        "short":    "qwen25_7b",
        "use_4bit": True,
        "backend":  "hf",
    },
    "deepseek-r1": {
        "hf_name":        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "short":          "deepseek_r1_8b",
        "use_4bit":       True,
        "backend":        "hf",
        "max_new_tokens": 300,  # R1 génère un bloc <think> avant de répondre
    },
    "mistral": {
        "hf_name":  "mistralai/Mistral-7B-Instruct-v0.3",
        "short":    "mistral_7b",
        "use_4bit": True,
        "backend":  "hf",
    },
    "magistral": {
        "hf_name":  "mistralai/Magistral-Small-24B-Instruct-2506",
        "short":    "magistral_24b",
        "use_4bit": True,
        "backend":  "hf",
    },
    # ── OpenAI API ────────────────────────────────────────
    "gpt-4o-mini": {
        "hf_name":  "gpt-4o-mini",
        "short":    "gpt4o_mini",
        "use_4bit": False,
        "backend":  "openai",
    },
    "gpt-4o": {
        "hf_name":  "gpt-4o",
        "short":    "gpt4o",
        "use_4bit": False,
        "backend":  "openai",
    },
    # ── Google (open-weights, HuggingFace) ───────────────
    "gemma": {
        "hf_name":  "google/gemma-2-9b-it",
        "short":    "gemma2_9b",
        "use_4bit": True,
        "backend":  "hf",
    },
    # ── Français / Multilingue ────────────────────────────
    "lucie": {
        "hf_name":  "OpenLLM-France/Lucie-7B-Instruct-v1.1",
        "short":    "lucie_7b",
        "use_4bit": True,
        "backend":  "hf",
    },
    "aya": {
        "hf_name":  "CohereForAI/aya-expanse-8b",
        "short":    "aya_expanse_8b",
        "use_4bit": True,
        "backend":  "hf",
    },
}

# ─────────────────────────────────────────────────────────
# 2. LABELS
# ─────────────────────────────────────────────────────────

LABEL_NAMES_3 = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_NAMES_2 = {0: "non-contradiction", 1: "contradiction"}

LABEL_ALIASES = {
    "entailment": 0, "entail": 0, "oui": 0, "vrai": 0, "yes": 0,
    "neutral": 1, "neutre": 1, "indeterminate": 1,
    "contradiction": 2, "contradicts": 2, "non": 2, "faux": 2, "no": 2,
    "non-contradiction": 0,
}

def parse_label(text: str, num_labels: int) -> int:
    """Extrait le label prédit depuis le texte généré."""
    # DeepSeek-R1 : ignorer le bloc <think>...</think>, garder ce qui suit
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text_low = text.lower()

    # Chercher la ligne "Label :" ou "Étiquette :" ou "Réponse :"
    for line in text_low.split("\n"):
        for kw in ["label :", "label:", "étiquette :", "réponse :", "answer:"]:
            if kw in line:
                remainder = line.split(kw, 1)[-1].strip()
                for alias, val in LABEL_ALIASES.items():
                    if alias in remainder:
                        if num_labels == 2 and val == 2:
                            return 1  # contradiction → 1 en mode binaire
                        if num_labels == 2 and val == 1:
                            return 0  # neutral → 0 en mode binaire
                        return val

    # Fallback : chercher le premier mot-clé dans tout le texte
    for alias, val in LABEL_ALIASES.items():
        if alias in text_low:
            if num_labels == 2 and val == 2:
                return 1
            if num_labels == 2 and val == 1:
                return 0
            return val

    return -1  # Pas trouvé


# ─────────────────────────────────────────────────────────
# 3. CHARGEMENT DATASETS
# ─────────────────────────────────────────────────────────

LABEL_MAP = {"yes": 0, "entailment": 0, "neutral": 1, "no": 2, "contradiction": 2}
INVALID = -1

def map_label(label):
    if isinstance(label, int):
        return label if label in [0, 1, 2] else INVALID
    s = str(label).lower().strip()
    if s in ("unknown", "undef"):
        return INVALID
    # Support for string integers "0", "1", "2"
    if s in ("0", "1", "2"):
        return int(s)
    return LABEL_MAP.get(s, INVALID)


def load_fracas_gq():
    ds = load_dataset("maximoss/fracas")["train"]
    ds = ds.filter(lambda x: x["topic"] == "GENERALIZED QUANTIFIERS"
                   and str(x["label"]).strip().lower() != "undef")
    def convert(ex):
        return {"premise": ex["premises"], "hypothesis": ex["hypothesis"],
                "label": map_label(ex["label"])}
    ds = ds.map(convert, remove_columns=ds.column_names)
    return ds.filter(lambda x: x["label"] != INVALID)


def load_gqnli():
    ds = load_dataset("maximoss/gqnli-fr")["test"]
    def convert(ex):
        return {"premise": ex["premise"], "hypothesis": ex["hypothesis"],
                "label": map_label(ex["label"])}
    ds = ds.map(convert, remove_columns=ds.column_names)
    return ds.filter(lambda x: x["label"] != INVALID)


def load_sick():
    raw = load_dataset("maximoss/sick-fr")
    result = {}
    for split, ds in raw.items():
        def convert(ex):
            lbl = str(ex["entailment_label"]).strip().upper()
            label = 0 if lbl == "ENTAILMENT" else (1 if lbl == "NEUTRAL" else 2)
            return {"premise": ex["sentence_A"], "hypothesis": ex["sentence_B"], "label": label}
        result[split] = ds.map(convert, remove_columns=ds.column_names)
    return result


def load_rte3():
    raw = load_dataset("maximoss/rte3-french")
    dev_rows, prem_groups = [], defaultdict(list)
    dev_keys = [k for k in raw.keys() if k != "test"]
    for k in dev_keys:
        for ex in raw[k]:
            idx = len(dev_rows)
            label = map_label(ex["label"])
            dev_rows.append({"premise": ex["premise"], "hypothesis": ex["hypothesis"], "label": label})
            prem_groups[ex["premise"]].append(idx)
    groups = list(prem_groups.values())
    random.shuffle(groups)
    n_train = int(len(dev_rows) * 0.8)
    train_idx, val_idx = [], []
    for g in groups:
        (train_idx if len(train_idx) < n_train else val_idx).extend(g)
    train_ds = Dataset.from_list([dev_rows[i] for i in train_idx]).filter(lambda x: x["label"] != INVALID)
    test_rows = [{"premise": ex["premise"], "hypothesis": ex["hypothesis"],
                  "label": map_label(ex["label"])} for ex in raw["test"]]
    test_ds = Dataset.from_list(test_rows).filter(lambda x: x["label"] != INVALID)
    return train_ds, test_ds


def load_daccord():
    ds = load_dataset("maximoss/daccord-contradictions")["train"]
    cols = [c for c in ds.column_names if c not in {"premise", "hypothesis", "label"}]
    ds = ds.remove_columns(cols)
    # Labels natifs : 0=compatible, 1=contradiction → déjà binaires
    return ds


DATASETS = {
    "fracas-gq": "fracas_gq",
    "gqnli":     "gqnli",
    "sick":      "sick",
    "rte3":      "rte3",
    "daccord":   "daccord",
}


def get_train_test(dataset_name: str):
    """Retourne (train_ds, test_ds, num_labels)."""
    if dataset_name == "fracas-gq":
        ds = load_fracas_gq()
        # Split 80/20 par index
        n = len(ds)
        train_ds = ds.select(range(int(n * 0.8)))
        test_ds  = ds.select(range(int(n * 0.8), n))
        return train_ds, test_ds, 3

    elif dataset_name == "gqnli":
        ds = load_gqnli()
        train_idx = list(range(80,100)) + list(range(180,200)) + list(range(280,300))
        test_idx  = list(range(300))
        return ds.select(train_idx), ds.select(test_idx), 3

    elif dataset_name == "sick":
        splits = load_sick()
        return splits["train"], splits["test"], 3

    elif dataset_name == "rte3":
        train_ds, test_ds = load_rte3()
        return train_ds, test_ds, 3

    elif dataset_name == "daccord":
        ds = load_daccord()
        n = len(ds)
        train_ds = ds.select(range(int(n * 0.5)))
        test_ds  = ds.select(range(int(n * 0.5), n))
        return train_ds, test_ds, 2

    raise ValueError(f"Dataset inconnu : {dataset_name}")


# ─────────────────────────────────────────────────────────
# 4. SÉLECTION DES EXEMPLES FEW-SHOT (équilibrés)
# ─────────────────────────────────────────────────────────

def load_fewshot_from_json(path: str, n_shots: int) -> list:
    """Charge les exemples few-shot depuis un fichier JSON manuel."""
    if n_shots == 0:
        return []
    with open(path, encoding="utf-8") as f:
        examples = json.load(f)
    return examples[:n_shots]


def select_fewshot_examples(train_ds, n_shots: int, num_labels: int, seed: int = 42) -> list:
    """Sélectionne n_shots exemples équilibrés depuis le train set."""
    if n_shots == 0:
        return []
    rng = random.Random(seed)
    per_class = max(1, n_shots // num_labels)
    by_label = defaultdict(list)
    for ex in train_ds:
        by_label[ex["label"]].append(ex)
    selected = []
    for label in range(num_labels):
        pool = by_label.get(label, [])
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
    rng.shuffle(selected)
    return selected[:n_shots]


def balanced_eval_sample(test_ds, max_samples: int, num_labels: int, seed: int = 42) -> Dataset:
    """Sous-échantillonne le test set de façon équilibrée si trop grand."""
    if len(test_ds) <= max_samples:
        return test_ds
    rng = random.Random(seed)
    per_class = max_samples // num_labels
    by_label = defaultdict(list)
    for i, ex in enumerate(test_ds):
        by_label[ex["label"]].append(i)
    selected = []
    for label in range(num_labels):
        pool = by_label.get(label, [])
        rng.shuffle(pool)
        selected.extend(pool[:per_class])
    rng.shuffle(selected)
    return test_ds.select(sorted(selected))


# ─────────────────────────────────────────────────────────
# 5. CONSTRUCTION DU PROMPT
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en inférence de langue naturelle (NLI) en français.
Ta tâche est de déterminer la relation logique entre une prémisse et une hypothèse.
Les labels possibles sont :
- entailment : l'hypothèse découle logiquement de la prémisse
- neutral : l'hypothèse n'est ni confirmée ni contredite par la prémisse
- contradiction : l'hypothèse contredit la prémisse

Réponds toujours en suivant ce format (label en premier, raisonnement en une phrase maximum) :
Label : <entailment | neutral | contradiction>
Raisonnement : <une phrase d'analyse>"""

SYSTEM_PROMPT_BINARY = """Tu es un expert en détection de contradictions en français.
Ta tâche est de déterminer si une hypothèse contredit une prémisse.
Les labels possibles sont :
- non-contradiction : l'hypothèse ne contredit pas la prémisse
- contradiction : l'hypothèse contredit la prémisse

Réponds toujours en suivant ce format (label en premier, raisonnement en une phrase maximum) :
Label : <non-contradiction | contradiction>
Raisonnement : <une phrase d'analyse>"""

SYSTEM_PROMPT_LABEL_ONLY = """Tu es un expert en inférence de langue naturelle (NLI) en français.
Ta tâche est de déterminer la relation logique entre une prémisse et une hypothèse.
Réponds uniquement avec le label, sans explication :
- entailment
- neutral
- contradiction"""

SYSTEM_PROMPT_LABEL_ONLY_BINARY = """Tu es un expert en détection de contradictions en français.
Réponds uniquement avec le label, sans explication :
- non-contradiction
- contradiction"""


def format_example(ex: dict, num_labels: int, with_answer: bool = True, use_cot: bool = True) -> str:
    label_names = LABEL_NAMES_2 if num_labels == 2 else LABEL_NAMES_3
    text = f"Prémisse : {ex['premise']}\nHypothèse : {ex['hypothesis']}"
    if with_answer:
        label_str = label_names.get(ex["label"], "unknown")
        if use_cot:
            cot = ex.get("chain_of_thought", "").strip()
            reasoning = cot if cot and cot != "À remplir" \
                else f"Cette relation est de type {label_str} car la prémisse et l'hypothèse sont analysées ensemble."
            text += f"\nLabel : {label_str}\nRaisonnement : {reasoning}"
        else:
            text += f"\nLabel : {label_str}"
    else:
        text += "\nLabel :" if use_cot else "\nLabel :"
    return text


def build_prompt(fewshot_examples: list, test_example: dict, num_labels: int, use_cot: bool) -> list:
    """Construit le prompt au format chat (liste de messages)."""
    if use_cot:
        system = SYSTEM_PROMPT_BINARY if num_labels == 2 else SYSTEM_PROMPT
    else:
        system = SYSTEM_PROMPT_LABEL_ONLY_BINARY if num_labels == 2 else SYSTEM_PROMPT_LABEL_ONLY
    messages = [{"role": "system", "content": system}]

    if fewshot_examples:
        examples_text = "\n\n---\n\n".join(
            format_example(ex, num_labels, with_answer=True, use_cot=use_cot)
            for ex in fewshot_examples
        )
        messages.append({
            "role": "user",
            "content": f"Voici {len(fewshot_examples)} exemple(s) :\n\n{examples_text}\n\n---\n\nMaintenant, analyse cet exemple :\n\n{format_example(test_example, num_labels, with_answer=False, use_cot=use_cot)}"
        })
    else:
        messages.append({
            "role": "user",
            "content": format_example(test_example, num_labels, with_answer=False, use_cot=use_cot)
        })

    return messages


# ─────────────────────────────────────────────────────────
# 6. INFÉRENCE
# ─────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_cfg: dict):
    """
    Retourne (model, tokenizer) pour les modèles HuggingFace,
    ou ((backend, client, model_name), None) pour les APIs externes.
    """
    backend = model_cfg.get("backend", "hf")
    model_name = model_cfg["hf_name"]
    print(f"Chargement de {model_name} (backend={backend})...")

    if backend == "openai":
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return ("openai", client, model_name), None

    # ── HuggingFace ──────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_cfg["use_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages: list, max_new_tokens: int = 150) -> str:
    """Génère une réponse pour un seul exemple (API ou HF)."""
    return batch_generate_responses(model, tokenizer, [messages], max_new_tokens)[0]


def batch_generate_responses(model, tokenizer, messages_list: list, max_new_tokens: int = 150) -> list:
    """Génère des réponses pour un batch d'exemples. Dispatche selon le backend."""

    # ── APIs externes : toujours séquentiel ──────────────
    if isinstance(model, tuple):
        backend, client, model_name = model
        results = []
        for messages in messages_list:
            if backend == "openai":
                import time
                response = ""
                for attempt in range(5):
                    try:
                        resp = client.chat.completions.create(
                            model=model_name, messages=messages,
                            max_tokens=max_new_tokens, temperature=0,
                        )
                        response = resp.choices[0].message.content
                        break
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "rate_limit" in err.lower():
                            wait = 30 * (2 ** attempt)
                            print(f"\n  [OpenAI] Rate limit — attente {wait}s...")
                            time.sleep(wait)
                        else:
                            print(f"\n  [OpenAI] Erreur : {e}")
                            break
                results.append(response)
        return results

    # ── HuggingFace : inférence batched ──────────────────
    # Construire les prompts texte (tokenize=False pour batch-tokenizer ensuite)
    prompts = []
    for messages in messages_list:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            prompt = "\n".join(m["content"] for m in messages) + "\n"
        prompts.append(prompt)

    # Left-padding obligatoire pour la génération batched sur modèles causaux
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=4096
    ).to(model.device)
    tokenizer.padding_side = orig_padding_side

    input_len = inputs["input_ids"].shape[-1]
    print(f"  [tokens] prompt={input_len} tokens (max_length=2048, max_new={max_new_tokens})")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True)
        for out in outputs
    ]


# ─────────────────────────────────────────────────────────
# 7. MÉTRIQUES
# ─────────────────────────────────────────────────────────

def compute_and_log_metrics(labels_true, labels_pred, num_labels, prefix="test"):
    valid_mask = [i for i, p in enumerate(labels_pred) if p != -1]
    if not valid_mask:
        print("ERREUR : aucune prédiction valide extraite.")
        return {}

    y_true = [labels_true[i] for i in valid_mask]
    y_pred = [labels_pred[i] for i in valid_mask]
    parse_rate = len(valid_mask) / len(labels_true)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    label_ids   = list(range(num_labels))
    label_names = [LABEL_NAMES_2[i] for i in label_ids] if num_labels == 2 \
                  else [LABEL_NAMES_3[i] for i in label_ids]

    prec, rec, f1_per, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    metrics = {
        f"{prefix}/accuracy":   acc,
        f"{prefix}/f1_macro":   f1,
        f"{prefix}/parse_rate": parse_rate,
    }
    for i, name in enumerate(label_names):
        metrics[f"{prefix}/{name}_f1"]       = f1_per[i]
        metrics[f"{prefix}/{name}_precision"] = prec[i]
        metrics[f"{prefix}/{name}_recall"]    = rec[i]

    wandb.log(metrics)
    wandb.log({f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, y_true=y_true, preds=y_pred, class_names=label_names
    )})
    wandb.summary.update({k: v for k, v in metrics.items()})

    print(f"\n{'='*50}")
    print(f"Accuracy : {acc:.2%} | F1 macro : {f1:.4f} | Parse rate : {parse_rate:.2%}")
    for i, name in enumerate(label_names):
        print(f"  {name:20s} F1={f1_per[i]:.3f} P={prec[i]:.3f} R={rec[i]:.3f} (n={sup[i]})")
    print(f"Matrice de confusion :\n{cm}")
    return metrics


# ─────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────

SHOTS_SWEEP_VALUES = [0, 1, 3, 5, 10]

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "test/f1_macro", "goal": "maximize"},
    "parameters": {
        "n_shots": {"values": SHOTS_SWEEP_VALUES},
    },
}

# Variables globales alimentées avant le lancement de l'agent WandB
_G_MODEL       = None
_G_TOKENIZER   = None
_G_MODEL_CFG   = None
_G_TRAIN_DS    = None
_G_TEST_DS     = None
_G_NUM_LABELS  = None
_G_ARGS        = None


def eval_run():
    """Fonction appelée par wandb.agent — lit n_shots depuis wandb.config."""
    run = wandb.init()
    config = wandb.config
    n_shots = config.n_shots
    use_cot = not _G_ARGS.no_cot

    run_name = f"{_G_MODEL_CFG['short']}_{DATASETS[_G_ARGS.dataset]}_n{n_shots}_{'cot' if use_cot else 'nocot'}"
    run.name = run_name
    run.config.update({
        "model":            _G_MODEL_CFG["hf_name"],
        "model_short":      _G_MODEL_CFG["short"],
        "dataset":          _G_ARGS.dataset,
        "use_cot":          use_cot,
        "num_labels":       _G_NUM_LABELS,
        "max_eval_samples": _G_ARGS.max_eval_samples,
        "seed":             _G_ARGS.seed,
    })

    # Sélection des exemples few-shot
    if _G_ARGS.fewshot_file:
        fewshot_examples = load_fewshot_from_json(_G_ARGS.fewshot_file, n_shots)
        print(f"Few-shot : {len(fewshot_examples)} exemples chargés depuis {_G_ARGS.fewshot_file}")
    else:
        fewshot_examples = select_fewshot_examples(_G_TRAIN_DS, n_shots, _G_NUM_LABELS, _G_ARGS.seed)
        print(f"Few-shot : {len(fewshot_examples)} exemples sélectionnés depuis le train set")

    # ── Inférence batched ──
    labels_true, labels_pred, raw_outputs = [], [], []
    total = len(_G_TEST_DS)
    batch_size = _G_ARGS.batch_size
    examples = list(_G_TEST_DS)

    print(f"\nInférence sur {total} exemples ({n_shots}-shot, batch_size={batch_size})...\n")
    for batch_start in range(0, total, batch_size):
        batch = examples[batch_start:batch_start + batch_size]
        messages_batch = [build_prompt(fewshot_examples, ex, _G_NUM_LABELS, use_cot) for ex in batch]
        model_max_tokens = _G_MODEL_CFG.get("max_new_tokens", _G_ARGS.max_new_tokens)
        responses = batch_generate_responses(_G_MODEL, _G_TOKENIZER, messages_batch, model_max_tokens)

        for ex, response in zip(batch, responses):
            predicted = parse_label(response, _G_NUM_LABELS)
            labels_true.append(ex["label"])
            labels_pred.append(predicted)
            raw_outputs.append({"premise": ex["premise"], "hypothesis": ex["hypothesis"],
                                "true": ex["label"], "pred": predicted, "response": response})

        done = min(batch_start + batch_size, total)
        if done % 20 == 0 or done == total:
            so_far_valid = [p for p in labels_pred if p != -1]
            parsed = len(so_far_valid)
            print(f"  [{done}/{total}] parse_rate={parsed/done:.0%}", end="")
            if parsed > 0:
                y_t = [labels_true[j] for j, p in enumerate(labels_pred) if p != -1]
                y_p = [p for p in labels_pred if p != -1]
                f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
                print(f"  F1={f1:.3f}", end="")
            print()

    # ── Métriques finales ──
    compute_and_log_metrics(labels_true, labels_pred, _G_NUM_LABELS)

    label_names = LABEL_NAMES_2 if _G_NUM_LABELS == 2 else LABEL_NAMES_3
    table = wandb.Table(columns=["premise", "hypothesis", "true_label", "pred_label", "response"])
    for r in raw_outputs[:50]:
        table.add_data(
            r["premise"], r["hypothesis"],
            label_names.get(r["true"], "?"),
            label_names.get(r["pred"], "invalid"),
            r["response"][:500]
        )
    wandb.log({"predictions_sample": table})
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           required=True, choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--dataset",         required=True, choices=list(DATASETS.keys()))
    p.add_argument("--n_shots",         type=int, default=5,
                   help="Nombre de shots (ignoré si --sweep)")
    p.add_argument("--sweep",           action="store_true",
                   help=f"Lance un WandB sweep sur n_shots ∈ {SHOTS_SWEEP_VALUES}")
    p.add_argument("--no_cot",          action="store_true", help="Désactive le CoT")
    p.add_argument("--max_eval_samples",type=int, default=300,
                   help="Nb max d'exemples de test évalués (0 = tous)")
    p.add_argument("--max_new_tokens",  type=int, default=60)
    p.add_argument("--batch_size",      type=int, default=4,
                   help="Nb d'exemples traités en parallèle sur le GPU")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--fewshot_file",    type=str, default=None,
                   help="JSON d'exemples few-shot manuels (fewshot_examples/<dataset>.json)")
    p.add_argument("--auto",            action="store_true", help="Pas de confirmation interactive")
    return p.parse_args()


def main():
    global _G_MODEL, _G_TOKENIZER, _G_MODEL_CFG, _G_TRAIN_DS, _G_TEST_DS, _G_NUM_LABELS, _G_ARGS

    args = parse_args()
    random.seed(args.seed)
    # En mode label-only, 20 tokens suffisent — ajustement automatique
    if args.no_cot and args.max_new_tokens == 60:
        args.max_new_tokens = 20
    _G_ARGS = args

    model_cfg = MODEL_CONFIGS[args.model]
    _G_MODEL_CFG = model_cfg
    use_cot = not args.no_cot

    n_runs = len(SHOTS_SWEEP_VALUES) if args.sweep else 1
    shots_display = SHOTS_SWEEP_VALUES if args.sweep else [args.n_shots]

    print(f"\n{'='*60}")
    print(f"Modèle  : {model_cfg['hf_name']}")
    print(f"Dataset : {args.dataset}")
    print(f"Shots   : {shots_display}")
    print(f"CoT     : {use_cot}")
    print(f"Runs    : {n_runs}")
    print(f"{'='*60}\n")

    if not args.auto:
        confirm = input("Lancer ? (o/n) : ").strip().lower()
        if confirm != "o":
            sys.exit(0)

    # ── Chargement données (une seule fois) ──
    _G_TRAIN_DS, _G_TEST_DS, _G_NUM_LABELS = get_train_test(args.dataset)
    print(f"Train : {len(_G_TRAIN_DS)} ex | Test : {len(_G_TEST_DS)} ex | Labels : {_G_NUM_LABELS}")

    if args.max_eval_samples > 0 and len(_G_TEST_DS) > args.max_eval_samples:
        _G_TEST_DS = balanced_eval_sample(_G_TEST_DS, args.max_eval_samples, _G_NUM_LABELS, args.seed)
        print(f"Test réduit à {len(_G_TEST_DS)} ex (équilibré)")

    # ── Chargement modèle (une seule fois) ──
    _G_MODEL, _G_TOKENIZER = load_model_and_tokenizer(model_cfg)

    if args.sweep:
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="fewshot-nli-fr")
        print(f"Sweep ID : {sweep_id}")
        wandb.agent(sweep_id, function=eval_run, count=n_runs)
    else:
        # Run unique — on injecte n_shots directement via une config minimale
        SWEEP_CONFIG_SINGLE = {
            "method": "grid",
            "metric": {"name": "test/f1_macro", "goal": "maximize"},
            "parameters": {"n_shots": {"values": [args.n_shots]}},
        }
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG_SINGLE, project="fewshot-nli-fr")
        wandb.agent(sweep_id, function=eval_run, count=1)

    print("\nTerminé !")


if __name__ == "__main__":
    main()
