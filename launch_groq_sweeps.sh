#!/bin/bash
# Lance les sweeps Groq séquentiellement (llama3-70b → qwen3-32b → gpt-oss-120b)
# Usage : bash launch_groq_sweeps.sh

export GROQ_API_KEY="GROQ_API_KEY_REMOVED"

SCRIPT="experiments/fewshot_cot/run_fewshot_cot.py"
FEWSHOT="experiments/fewshot_cot/fewshot_examples/gqnli.json"
DATASET="gqnli"

echo "=== Sweeps Groq séquentiels ==="

for MODEL in "llama3-70b" "qwen3-32b" "gpt-oss-120b"; do
    echo ""
    echo ">>> Lancement : $MODEL"
    python3 $SCRIPT --model $MODEL --dataset $DATASET --sweep --auto --fewshot_file $FEWSHOT
    echo ">>> Terminé : $MODEL"
done

echo ""
echo "=== Tous les sweeps terminés ==="
