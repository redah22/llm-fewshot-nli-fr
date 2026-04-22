"""
Script de diagnostic LÉGER (sans dépendances HuggingFace).
Télécharge les fichiers TSV bruts et vérifie la cohérence des labels.
"""
import csv
import io
import urllib.request
from collections import Counter

# ── 1. Téléchargement des données brutes ─────────────────────────────

def download_tsv(url):
    """Télécharge un fichier TSV et retourne les lignes comme liste de dicts."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    return list(reader)

# ── 2. Normalisation (copie exacte de train_prefix_tuning.py) ────────

def normalize_label(label):
    LABEL_MAP = {
        "yes": "vrai", "entailment": "vrai", 0: "vrai", "0": "vrai",
        "unknown": "neutre", "undef": "neutre", "neutral": "neutre", 1: "neutre", "1": "neutre",
        "no": "faux", "contradiction": "faux", 2: "faux", "2": "faux"
    }
    s = str(label).lower().strip()
    return LABEL_MAP.get(s, "neutre")

# ── 3. Vérification GQNLI-FR ────────────────────────────────────────

def verify_gqnli():
    url = "https://huggingface.co/datasets/maximoss/gqnli-fr/resolve/main/gqnli.test.fr.tsv"
    print("\n📥 Téléchargement de GQNLI-FR...")
    rows = download_tsv(url)
    print(f"  Total lignes dans le fichier : {len(rows)}")
    print(f"  Colonnes : {list(rows[0].keys())}")

    # Indices utilisés par notre code
    train_idx = list(range(0, 60)) + list(range(100, 160)) + list(range(200, 260))
    val_idx   = list(range(60, 80)) + list(range(160, 180)) + list(range(260, 280))

    for split_name, indices in [("train", train_idx), ("validation", val_idx)]:
        selected = [rows[i] for i in indices]
        
        # Trouver la colonne label
        label_col = None
        for col in rows[0].keys():
            if "label" in col.lower():
                label_col = col
                break
        
        raw_labels = [r[label_col] for r in selected]
        norm_labels = [normalize_label(l) for l in raw_labels]
        dist = Counter(norm_labels)
        raw_unique = sorted(set(raw_labels))

        print(f"\n{'='*60}")
        print(f"  GQNLI-FR / {split_name}  ({len(selected)} exemples)")
        print(f"{'='*60}")
        print(f"  Distribution normalisée : vrai={dist.get('vrai',0)}  neutre={dist.get('neutre',0)}  faux={dist.get('faux',0)}")
        print(f"  Labels bruts uniques : {raw_unique}")
        for r in raw_unique:
            print(f"    '{r}' → '{normalize_label(r)}'")

        # Trouver les colonnes premise/hypothesis
        premise_col = None
        hyp_col = None
        for col in rows[0].keys():
            cl = col.lower()
            if "premise" in cl or "premi" in cl:
                premise_col = col
            if "hypo" in cl:
                hyp_col = col

        print(f"\n  Colonnes utilisées : premise='{premise_col}', hypothesis='{hyp_col}'")
        print(f"\n  --- 3 premiers exemples (split {split_name}) ---")
        for i in range(min(3, len(selected))):
            ex = selected[i]
            p = (ex.get(premise_col, "N/A") or "N/A")[:100]
            h = (ex.get(hyp_col, "N/A") or "N/A")[:100]
            print(f"  [{i}] Prémisse : {p}")
            print(f"       Hypothèse: {h}")
            print(f"       Label brut: '{ex[label_col]}'  →  normalisé: '{normalize_label(ex[label_col])}'")

# ── 4. Vérification FraCaS ──────────────────────────────────────────

def verify_fracas():
    url = "https://huggingface.co/datasets/maximoss/fracas/resolve/main/fracas_in_english_and_french.tsv"
    print("\n\n📥 Téléchargement de FraCaS...")
    rows = download_tsv(url)
    print(f"  Total lignes dans le fichier : {len(rows)}")
    print(f"  Colonnes : {list(rows[0].keys())}")

    # On prend les 75 premières lignes (indices 0-74)
    selected = rows[:75]
    
    # Trouver la colonne label
    label_col = None
    for col in rows[0].keys():
        if "label" in col.lower() or "answer" in col.lower():
            label_col = col
            break

    raw_labels = [r[label_col] for r in selected]
    norm_labels = [normalize_label(l) for l in raw_labels]
    dist = Counter(norm_labels)
    raw_unique = sorted(set(raw_labels))

    print(f"\n{'='*60}")
    print(f"  FraCaS / 0-74  ({len(selected)} exemples)")
    print(f"{'='*60}")
    print(f"  Distribution normalisée : vrai={dist.get('vrai',0)}  neutre={dist.get('neutre',0)}  faux={dist.get('faux',0)}")
    print(f"  Labels bruts uniques : {raw_unique}")
    for r in raw_unique:
        print(f"    '{r}' → '{normalize_label(r)}'")

    # Trouver les colonnes premise/hypothesis
    premise_col = None
    hyp_col = None
    for col in rows[0].keys():
        cl = col.lower()
        if "premise" in cl or "premi" in cl:
            premise_col = col
        if "hypo" in cl:
            hyp_col = col

    print(f"\n  Colonnes utilisées : premise='{premise_col}', hypothesis='{hyp_col}'")
    print(f"\n  --- 5 premiers exemples ---")
    for i in range(min(5, len(selected))):
        ex = selected[i]
        p = (ex.get(premise_col, "N/A") or "N/A")[:100]
        h = (ex.get(hyp_col, "N/A") or "N/A")[:100]
        print(f"  [{i}] Prémisse : {p}")
        print(f"       Hypothèse: {h}")
        print(f"       Label brut: '{ex[label_col]}'  →  normalisé: '{normalize_label(ex[label_col])}'")


# ── MAIN ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔍 VÉRIFICATION DE COHÉRENCE DU PIPELINE PREFIX-TUNING")
    print("   (Sans dépendances HuggingFace — lecture directe des TSV)")

    verify_gqnli()
    verify_fracas()

    print("\n\n✅ Vérification terminée.")
