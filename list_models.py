"""List all available models from your Ollama service."""
import os
from dotenv import load_dotenv
import requests

load_dotenv()

base_url = os.getenv("OLLAMA_BASE_URL")
api_key = os.getenv("OLLAMA_API_KEY")

headers = {
    'Authorization': f'Bearer {api_key}'
} if api_key else {}

print("=" * 60)
print("Available Models on Your Ollama Service")
print("=" * 60)

try:
    response = requests.get(f"{base_url}/api/tags", headers=headers, timeout=10)

    if response.status_code == 200:
        data = response.json()
        models = data.get('models', [])

        print(f"\nFound {len(models)} models:\n")

        for i, model in enumerate(models, 1):
            name = model.get('name', 'Unknown')
            size_gb = model.get('size', 0) / (1024**3)  # Convert to GB
            modified = model.get('modified_at', 'Unknown')

            print(f"{i}. {name}")
            print(f"   Size: {size_gb:.2f} GB")
            print(f"   Modified: {modified}")
            print()

        print("=" * 60)
        print("To use one of these models:")
        print("1. Update your .env file")
        print("2. Change OLLAMA_MODEL to one of the model names above")
        print("   Example: OLLAMA_MODEL=glm-4.7")
        print("=" * 60)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"Error: {e}")
