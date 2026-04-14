import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_MODEL = "almanach/camembert-base"
LORA_PATH = "/kaggle/working/camembert_lora_final"

if not os.path.exists(LORA_PATH):
    LORA_PATH = "../../models/camembert_lora_final"
    if not os.path.exists(LORA_PATH):
        LORA_PATH = "./camembert_lora_final"

print("=====================================================")
print("  EVALUATION GLOBALE (SICK, FRACAS, GQNLI, RTE3)     ")
print("=====================================================\n")

print(f"1. Chargement du modèle de base ({BASE_MODEL}) silencieusement...")
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

print(f"2. Greffe du cerveau LoRA ({LORA_PATH})...")
try:
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
except Exception as e:
    print(f"❌ ERREUR: Impossible de trouver le modèle ou le tokenizer dans '{LORA_PATH}'")
    exit(1)

LABEL_MAP = {"yes": 0, "entailment": 0, "unknown": 1, "undef": 1, "neutral": 1, "no": 2, "contradiction": 2}
def map_label(label):
    if isinstance(label, int) and label in [0, 1, 2]: return label
    s = str(label).lower().strip()
    if s in LABEL_MAP: return LABEL_MAP[s]
    try: return int(s)
    except: return 1

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    return {"accuracy": acc, "cm": cm.tolist()}

trainer = Trainer(model=model, compute_metrics=compute_metrics)

all_results = {}
all_labels_global = []
all_preds_global = []

def eval_dataset(name, dataset, is_sick=False):
    print(f"\n▶️ TEST : {name}\n" + "-"*40)
    
    # Nettoyage global: enlever tout ce qui a un vrai label 'undef' textuel
    dataset = dataset.filter(lambda x: str(x.get('label', x.get('entailment_label', ''))).strip().lower() != "undef")
    
    if is_sick:
        def convert_sick(ex):
            return {'premise': ex['sentence_A'], 'hypothesis': ex['sentence_B'], 'label': map_label(ex['entailment_label'])}
        dataset = dataset.map(convert_sick, remove_columns=dataset.column_names)
        
        vrai = dataset.filter(lambda x: x['label'] == 0)
        neutre = dataset.filter(lambda x: x['label'] == 1)
        faux = dataset.filter(lambda x: x['label'] == 2)
        mc = min(len(vrai), len(neutre), len(faux))
        balanced = concatenate_datasets([vrai.select(range(mc)), neutre.select(range(mc)), faux.select(range(mc))]).shuffle(seed=42)
        
        total = len(balanced)
        dataset = balanced.select(range(int(total * 0.8), total))
    else:
        # Standardisation des colonnes pour les autres datasets
        def standardize_cols(ex):
            p = ex.get("premise", ex.get("sentence_A", ex.get("sentence1", "")))
            h = ex.get("hypothesis", ex.get("sentence_B", ex.get("sentence2", "")))
            raw_labels = ex.get("label", ex.get("entailment_label", 1))
            return {'premise': p, 'hypothesis': h, 'label': map_label(raw_labels)}
        dataset = dataset.map(standardize_cols, remove_columns=dataset.column_names)

    def tok_fn(ex):
        res = tokenizer(ex["premise"], ex["hypothesis"], truncation=True, padding="max_length", max_length=128)
        res["labels"] = ex["label"]
        return res

    tok_data = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names)
    tok_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    preds_output = trainer.predict(tok_data)
    acc = preds_output.metrics['test_accuracy']
    cm = preds_output.metrics['test_cm']
    
    preds = np.argmax(preds_output.predictions, axis=1)
    all_labels_global.extend(preds_output.label_ids)
    all_preds_global.extend(preds)
    
    print(f"🎯 Accuracy : {acc:.2%}")
    print(f"📊 Matrice de Confusion (0=Vrai, 1=Neutre, 2=Faux) :\n{np.array(cm)}")
    
    all_results[name] = {"acc": acc, "size": len(dataset)}


print("\n3. Lancement des évaluations...")

# 1. SICK-FR
sick = load_dataset('maximoss/sick-fr')
sick_ds = concatenate_datasets(list(sick.values())) if len(sick.keys()) > 1 else list(sick.values())[0]
eval_dataset("SICK-FR (Intra-Domain Équilibré)", sick_ds, is_sick=True)

# 2. FRACAS
fracas = load_dataset('maximoss/fracas')['train']
eval_dataset("FraCaS (Cross-Domain)", fracas)

# 3. GQNLI-FR
try:
    gqnli = load_dataset('maximoss/gqnli-fr')
    gqnli_ds = concatenate_datasets(list(gqnli.values())) if len(gqnli.keys()) > 1 else list(gqnli.values())[0]
    eval_dataset("GQNLI-FR (Cross-Domain)", gqnli_ds)
except Exception as e:
    print(f"Erreur GQNLI: {e}")

# 4. RTE3-French
try:
    rte3 = load_dataset('maximoss/rte3-french')
    rte3_ds = concatenate_datasets(list(rte3.values())) if len(rte3.keys()) > 1 else list(rte3.values())[0]
    eval_dataset("RTE3-French (Cross-Domain)", rte3_ds)
except Exception as e:
    print(f"Erreur RTE3: {e}")

print("\n" + "="*60)
print("             🏆 BILAN DES PERFORMANCES 🏆")
print("="*60)

for name, res in all_results.items():
    print(f" - {name:<35} : {res['acc']:.2%} ({res['size']} exemples)")

global_acc = accuracy_score(all_labels_global, all_preds_global)
print("-" * 60)
print(f" 🌍 ACCURACY GLOBALE MOYENNE : {global_acc:.2%} (sur {len(all_labels_global)} exemples totaux !)")
print("="*60)
