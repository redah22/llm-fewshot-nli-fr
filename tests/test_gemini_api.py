"""
Test rapide de la clé API Gemini
"""

from dotenv import load_dotenv
import os

# Charger .env
load_dotenv()

# Vérifier la clé
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("❌ GOOGLE_API_KEY non trouvée dans .env")
    exit(1)

print(f"✅ Clé Gemini trouvée: {api_key[:20]}...")

# Test avec Gemini
try:
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    print("\nTest d'appel API Gemini...")
    
    # Utiliser un modèle qui existe
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    response = model.generate_content("Réponds juste 'OK' si tu me reçois")
    
    result = response.text
    
    print(f"✅ Réponse de l'API: {result}")
    print(f"   Modèle utilisé: gemini-2.5-flash")
    
    print("\n🎉 API Gemini fonctionne!")
    
    # Tester aussi gemini-pro
    print("\nTest Gemini 2.5 Pro...")
    model_pro = genai.GenerativeModel('gemini-2.5-pro')
    response_pro = model_pro.generate_content("Réponds 'OK'")
    print(f"✅ Gemini 2.5 Pro: {response_pro.text}")
    
except ImportError:
    print("❌ Module google-generativeai non installé")
    print("   Lancez: pip install google-generativeai")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("\nVérifiez:")
    print("  1. Que la clé API est correcte")
    print("  2. Que vous avez accès à Gemini Pro")
