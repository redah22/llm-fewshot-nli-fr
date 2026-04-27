import os
os.environ["WANDB_DISABLED"] = "true"  # Désactive WandB pour ce run unique
import torch
import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_MODEL = "almanach/camembert-base"
OUTPUT_DIR = "/kaggle/working/camembert_lora_final"

print("=====================================================")
print("  ENTRAÎNEMENT DU MODÈLE FINAL CAMEMBERT (SICK-FR)   ")
print("=====================================================\n")

print("1. Chargement de CamemBERT et Tokenizer...")
global_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# --- PRÉPARATION DU DATASET ---
print("2. Préparation du Dataset équilibré SICK-FR...")
sick = load_dataset('maximoss/sick-fr')
if len(sick.keys()) > 1:
    data = concatenate_datasets(list(sick.values()))
else:
    data = list(sick.values())[0]

def convert_sick(ex):
    lbl = str(ex['entailment_label']).strip().upper()
    if lbl == 'ENTAILMENT': label_id = 0
    elif lbl == 'NEUTRAL': label_id = 1
    elif lbl == 'CONTRADICTION': label_id = 2
    else: label_id = 1
    return {'premise': ex['sentence_A'], 'hypothesis': ex['sentence_B'], 'label': label_id}

data = data.map(convert_sick, remove_columns=data.column_names)

vrai_pool = data.filter(lambda x: x['label'] == 0)
neutre_pool = data.filter(lambda x: x['label'] == 1)
faux_pool = data.filter(lambda x: x['label'] == 2)

max_per_class = min(len(vrai_pool), len(neutre_pool), len(faux_pool))

test_vrai = vrai_pool.select(range(max_per_class))
test_neutre = neutre_pool.select(range(max_per_class))
test_faux = faux_pool.select(range(max_per_class))

balanced_data = concatenate_datasets([test_vrai, test_neutre, test_faux]).shuffle(seed=42)

total = len(balanced_data)
train_size = int(total * 0.6)
val_size = int(total * 0.2)

dataset = DatasetDict({
    'train': balanced_data.select(range(0, train_size)),
    'validation': balanced_data.select(range(train_size, train_size + val_size)),
    'test': balanced_data.select(range(train_size + val_size, total))
})

# --- TOKENISATION ---
def tokenize_fn(examples):
    res = global_tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=128)
    res["labels"] = examples["label"]
    return res

train_data = dataset['train'].map(tokenize_fn, batched=True, remove_columns=dataset['train'].column_names)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

val_data = dataset['validation'].map(tokenize_fn, batched=True, remove_columns=dataset['validation'].column_names)
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_data = dataset['test'].map(tokenize_fn, batched=True, remove_columns=dataset['test'].column_names)
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    try:
        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
        print(f"\nMatrice de confusion:\n{cm}")
    except Exception: pass
    return {"accuracy": accuracy_score(labels, predictions)}

# --- MODÈLE ET LORA ---
print(f"\n3. Définition du Cerveau LoRA (Hyperparamètres optimisés)...")
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
    bias="none",
)
model = get_peft_model(base_model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
tot = sum(p.numel() for p in model.parameters())
print(f"Paramètres entraînables : {trainable:,} ({100 * trainable / tot:.2f}%)")

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",  
    save_strategy="no",     # On sauve le meilleur modèle tout à la fin
    learning_rate=0.0005,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_steps=10,
    dataloader_num_workers=0,  # Evite le deadlock
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

print("\n4. Lancement de l'Entraînement Final (20 epochs)...")
trainer.train()

print(f"\n5. Lancement de l'Évaluation sur {len(test_data)} exemples (SICK-FR Equilibré)...")
test_results = trainer.evaluate(test_data)
print(f"FINAL TEST ACCURACY : {test_results['eval_accuracy']:.2%}")

print(f"\n6. Sauvegarde Définitive dans {OUTPUT_DIR}...")
# Obligatoire : Sauvegarder explicitement les poids finaux pour le téléchargement
model.save_pretrained(OUTPUT_DIR)
global_tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ TERMINÉ ! Allez dans l'onglet 'Output' de Kaggle pour télécharger votre modèle !")
