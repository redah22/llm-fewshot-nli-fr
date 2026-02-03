# Prochaines Étapes - After Setup

Vous avez fait le setup ! Maintenant voilà ce qui reste à faire.

## Ce que vous avez

✅ FraCaS chargé et filtré sur GENERALIZED QUANTIFIERS  
✅ GQNLI-FR divisé en dev (240) et test (60)  
✅ Exemples few-shot préparés  
✅ Tout sauvegardé dans `data/processed/`

## Prochaines étapes

### Option 1: Évaluation avec LLM (GPT-4, Gemini, etc.)

**Si vous avez une API key:**

1. Configurer l'API key dans `.env`
2. Créer un script d'évaluation few-shot
3. Tester 0-shot vs 1-shot vs 3-shot vs 5-shot sur **GQNLI-FR dev**
4. Comparer les résultats
5. Évaluer sur test à la fin

Je peux vous créer ce script si vous voulez.

### Option 2: Évaluation avec modèle local (CamemBERT, etc.)

**Sans API key:**

1. Fine-tuner CamemBERT sur FraCaS GQ
2. Évaluer sur GQNLI-FR dev
3. Comparer avec baseline
4. Évaluer sur test à la fin

### Option 3: Analyse qualitative d'abord

**Pour comprendre le problème:**

1. Regarder manuellement quelques exemples
2. Identifier les difficultés
3. Analyser les différences FraCaS vs GQNLI-FR
4. Puis passer à l'évaluation automatique

## Qu'est-ce que vous voulez faire ?

**Dites-moi:**
1. Vous avez une API key pour GPT/Gemini ?
2. Ou vous préférez commencer avec des modèles locaux ?
3. Ou juste analyser les données d'abord ?

Je créerai le notebook/script adapté !
