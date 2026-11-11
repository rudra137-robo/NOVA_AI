from pathlib import Path
import sys

# Add the NOVA project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.tiny_model import generate_text, get_model_info

print("Checkpoint information:")
print(get_model_info())

print("\nGenerated text:")
print(generate_text("hello ", max_new_tokens=100))