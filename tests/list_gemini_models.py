"""
Liste les modèles Gemini disponibles avec votre clé API
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os

# Charger la clé depuis .env
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("❌ GOOGLE_API_KEY non trouvée")
    exit(1)

# Configurer
genai.configure(api_key=api_key)

print("Liste des modèles Gemini disponibles avec generateContent :\n")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ {m.name}")
            
except Exception as e:
    print(f"Erreur: {e}")
