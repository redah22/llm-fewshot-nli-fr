"""
Test rapide de la clé API OpenAI
"""

from dotenv import load_dotenv
import os

# Charger .env
load_dotenv()

# Vérifier la clé
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    print("❌ OPENAI_API_KEY non trouvée dans .env")
    exit(1)

print(f"✅ Clé OpenAI trouvée: {api_key[:20]}...")

# Test avec OpenAI
try:
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    print("\nTest d'appel API...")
    
    # Simple test
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Modèle pas cher pour tester
        messages=[
            {"role": "user", "content": "Réponds juste 'OK' si tu me reçois"}
        ],
        max_tokens=10
    )
    
    result = response.choices[0].message.content
    
    print(f"✅ Réponse de l'API: {result}")
    print(f"   Modèle utilisé: {response.model}")
    print(f"   Tokens: {response.usage.total_tokens}")
    
    print("\n🎉 API OpenAI fonctionne!")
    
    # Lister les modèles disponibles
    print("\n📋 Modèles GPT disponibles avec votre clé:")
    models = client.models.list()
    gpt_models = [m.id for m in models.data if 'gpt' in m.id.lower()]
    for model in sorted(gpt_models):
        print(f"  - {model}")
    
except ImportError:
    print("❌ Module openai non installé")
    print("   Lancez: pip install openai")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("\nVérifiez:")
    print("  1. Que vous avez des crédits sur votre compte OpenAI")
    print("  2. Que la clé API est correcte")
