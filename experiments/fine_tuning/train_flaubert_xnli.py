"""
Fine-tuning complet de FlauBERT sur XNLI (fr)

Ce script utilise le corpus XNLI-fr préparé via setup_xnli.py pour fine-tuner
le modèle flabert-base-cased. C'est une étape de pré-entraînement (Transfer Learning)
avant de fine-tuner sur des corpus plus petits comme GQNLI ou FraCaS.
"""

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score
import os
import sys

print("="*60)
print("FINE-TUNING FLAUBERT SUR XNLI-FR")
print("="*60)

# 1. Charger les données XNLI
data_path = "data/processed/xnli_fr"
if not os.path.exists(data_path):
    print(f"❌ Erreur: Le dataset XNLI introuvable à {data_path}")
    print("Veuillez d'abord exécuter: python3 experiments/data_utils/setup_xnli.py")
    sys.exit(1)

print("\nChargement du dataset XNLI...")
dataset = DatasetDict.load_from_disk(data_path)

# XNLI est grand (392k exemples train). On peut s'entraîner sur un sous-ensemble 
# pour que ça ne prenne pas des semaines sur un Mac.
print("\nQuelle taille d'entraînement souhaitez-vous ?")
print("1. Rapide (10 000 exemples) - Idéal pour vérifier que ça tourne")
print("2. Moyen (50 000 exemples) - Bon compromis")
print("3. Complet (392 702 exemples) - ⚠️ Très long (plusieurs jours/semaines sur Mac)")

choice = input("\nVotre choix (1, 2 ou 3): ").strip()

if choice == "1":
    train_size = 10000
elif choice == "2":
    train_size = 50000
elif choice == "3":
    train_size = len(dataset['train'])
else:
    print("Choix invalide, par défaut Moyen (50k)")
    train_size = 50000

train_data = dataset['train'].shuffle(seed=42).select(range(train_size))
eval_data = dataset['validation']  # 2490 examples
test_data = dataset['test']        # 5010 examples

print(f"\nDonnées sélectionnées :")
print(f"Train: {len(train_data)}")
print(f"Val: {len(eval_data)}")
print(f"Test: {len(test_data)}")


# 2. Charger Modèle et Tokenizer
model_name = 'flaubert/flaubert_base_cased'
print(f"\nChargement de {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# 3. Tokenization
def preprocess_function(examples):
    # XNLI utilises 'premise' and 'hypothesis' conceptually, but keys might differ slightly
    # In HF datasets xnli, the keys are usually 'premise' and 'hypothesis'
    
    result = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        padding='max_length',
        max_length=128,
        return_token_type_ids=True
    )
    
    # In XNLI, labels are already integers: 0 (entailment), 1 (neutral), 2 (contradiction)
    result['labels'] = examples['label']
    return result

print("\nTokenization en cours...")
tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
tokenized_eval = eval_data.map(preprocess_function, batched=True, remove_columns=eval_data.column_names)
tokenized_test = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'token_type_ids'])
tokenized_eval.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'token_type_ids'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'token_type_ids'])

# 4. Entraînement
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

output_dir = "checkpoints/flaubert_xnli_fr"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy='epoch', # Evaluate every epoch
    save_strategy='epoch', # Save every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3, # 3 epochs are usually enough for large NLI datasets
    weight_decay=0.01,
    load_best_model_at_end=True, # Will load the best model based on validation acc
    metric_for_best_model='accuracy',
    logging_dir="logs/flaubert_xnli_fr",
    logging_steps=100,
    report_to='none',
    remove_unused_columns=False,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("\nDébut du fine-tuning sur XNLI...")
trainer.train()

# 5. Sauvegarde Finale
final_model_dir = "models/flaubert_xnli_fr"
os.makedirs(final_model_dir, exist_ok=True)
print(f"\nSauvegarde du meilleur modèle dans {final_model_dir}")
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# 6. Evaluation sur le Test Set XNLI
print("\nÉvaluation finale sur le set de Test XNLI (5010 exemples)...")
trainer.eval_dataset = tokenized_test
test_results = trainer.evaluate()

print(f"\n✅ Terminé ! Accuracy sur XNLI-Test : {test_results['eval_accuracy']:.2%}")
print(f"Modèle disponible sous : {final_model_dir}")
print("Vous pouvez maintenant utiliser ce modèle comme point de départ dans eval_flaubert.py (Mode 2 -> charger ce chemin)")
