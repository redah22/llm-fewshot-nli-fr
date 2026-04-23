# TER M1 : Inférence en Langue Naturelle (NLI) en contexte Cross-Domain

Ce projet, réalisé dans le cadre d'un Master 1 (TER), a pour objectif d'étudier la capacité de généralisation des modèles de langage sur la tâche de **Natural Language Inference (NLI)** en français. L'inférence linguistique consiste à déterminer si une *hypothèse* est vraie (Entailment), fausse (Contradiction) ou neutre en se basant uniquement sur une *prémisse*.

Le cœur de notre travail explore comment différents types d'architectures s'adaptent lorsqu'ils sont confrontés à de nouvelles données inconnues de leur domaine d'entraînement initial (Cross-Domain) et à des distributions de classes imparfaitement équilibrées.

### Axes de recherche du projet :
1. **Généralisation Cross-Domain :** Entraînement sur des corpus formels/logiques (tels que *FraCaS* ou *RTE3*) et évaluation sur des corpus sémantiques ou du monde réel (*SICK-FR*, *GQNLI-FR*, *DACCORD*) pour observer la complaisance et le sur-apprentissage du modèle.
2. **Comparaison d'Architectures :** Alignement et comparaison des capacités de raisonnement entre trois grands paradigmes du Deep Learning :
   - Les **Encodeurs** (CamemBERT), réputés pour leur compréhension fine et chirurgicale.
   - Les **Encodeurs-Décodeurs** (T5 / Gemma Seq2Seq) avec l'intégration du Few-Shot Prompting (In-Context Learning).
   - Les **Décodeurs purs** (CroissantLLM), massifs en paramètres et dotés de fort bon sens interne, reconvertis exceptionnellement en classifieurs terminaux.
3. **Résilience Mathématique (Class Weighting) :** Recherche de la pénalité de perte optimale (`loss_penalty`) à attribuer aux réseaux de neurones pour compenser artificiellement un déséquilibre sévère des classes réelles lors d'évaluations très déséquilibrées (ex: accorder artificiellement plus de poids à la classe rare *Contradiction*).
