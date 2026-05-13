#!/bin/bash
# Push des kernels Kaggle via la CLI
#
# Prérequis :
#   pip install kaggle
#   Placer ~/.kaggle/kaggle.json (API key depuis kaggle.com/settings/api)
#
# Usage :
#   bash push_kernels.sh                          # push tous les kernels
#   bash push_kernels.sh camembert 1 4 7          # push camembert EXP 1,4,7
#   bash push_kernels.sh gpt2 8                   # push gpt2 EXP 8

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Filtrage par modèle et/ou expériences ────────────────────
MODEL_FILTER="${1:-}"
shift 2>/dev/null || true
EXP_FILTER=("$@")

push_kernel() {
    local dir="$1"
    echo "→ Push : $dir"
    kaggle kernels push -p "$SCRIPT_DIR/$dir"
    echo "  [OK] $dir"
}

# ── Parcours des dossiers de kernels ────────────────────────
for dir in "$SCRIPT_DIR"/*/; do
    dir_name="$(basename "$dir")"

    # Ignorer les non-kernels (pas de kernel-metadata.json)
    [[ ! -f "$dir/kernel-metadata.json" ]] && continue

    # Filtre modèle
    if [[ -n "$MODEL_FILTER" && "$dir_name" != "${MODEL_FILTER}"* ]]; then
        continue
    fi

    # Filtre expériences
    if [[ ${#EXP_FILTER[@]} -gt 0 ]]; then
        exp_num="${dir_name##*exp}"
        match=0
        for e in "${EXP_FILTER[@]}"; do
            [[ "$exp_num" == "$e" ]] && match=1 && break
        done
        [[ $match -eq 0 ]] && continue
    fi

    push_kernel "$dir_name"
done

echo ""
echo "Push terminé."
