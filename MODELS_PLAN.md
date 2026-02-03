# Plan d'Évaluation Multi-Modèles

Liste des modèles à tester demandée par le superviseur.

## Modèles LLM (API)

Ces modèles nécessitent des API keys:

1. **GPT-5** (OpenAI)
2. **Mistral 3** (Mistral AI)
3. **Gemini 3** (Google)
4. **DeepSeek Reasoning Model** (DeepSeek)
5. **Llama 4** (Meta - via API ou local)

## Modèles Encodeurs (Local/HuggingFace)

Ces modèles peuvent tourner localement:

1. **CamemBERT** - modèle français de base
2. **mDeBERTa** - multilingue, très performant
3. **XLM-R** (XLM-RoBERTa) - multilingue baseline
4. **FlauBERT** - français, alternative à CamemBERT
5. **CamemBERTa** - variante de CamemBERT

## Stratégie d'Évaluation

### LLMs (Few-Shot)
- 0-shot, 1-shot, 3-shot, 5-shot
- Exemples depuis FraCaS GQ
- Éval sur GQNLI-FR dev
- Choisir meilleur n_shots
- Test final sur GQNLI-FR test

### Encodeurs (Fine-Tuning)
- Fine-tune sur FraCaS GQ
- Éval sur GQNLI-FR dev
- Early stopping sur dev
- Test final sur GQNLI-FR test

## Configuration Nécessaire

### API Keys (.env)
```
OPENAI_API_KEY=...
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
LLAMA_API_KEY=...  # ou local
```

### Compute
- LLMs: API calls (coût à prévoir)
- Encodeurs: GPU conseillé pour fine-tuning

## Scripts à Créer

1. `scripts/eval_llm_few_shot.py` - Éval LLMs avec few-shot
2. `scripts/eval_encoders.py` - Fine-tune et éval encodeurs
3. `scripts/compare_all_models.py` - Comparer tous les résultats

## Timeline Estimé

- LLMs (few-shot): 1-2 jours (API calls)
- Encodeurs (fine-tuning): 3-5 jours (selon GPU)
- Analyse résultats: 2-3 jours

Total: ~1-2 semaines pour tout tester
