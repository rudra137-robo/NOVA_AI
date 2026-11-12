from pathlib import Path

import torch

from model_training.generate_from_checkpoint import TinyTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "model_training"
    / "tiny_transformer_best.pt"
)

_model = None
_char_to_id = None
_id_to_char = None
_checkpoint_info = None


def load_model():
    """Load the trained Transformer once and reuse it."""
    global _model, _char_to_id, _id_to_char, _checkpoint_info

    if _model is not None:
        return

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    characters = checkpoint["characters"]

    _char_to_id = {
        char: index for index, char in enumerate(characters)
    }
    _id_to_char = {
        index: char for index, char in enumerate(characters)
    }

    _model = TinyTransformer(len(characters))
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    _checkpoint_info = {
        "best_step": checkpoint.get("best_step"),
        "best_validation_loss": checkpoint.get(
            "best_validation_loss"
        ),
    }


def generate_text(prompt="hello ", max_new_tokens=100):
    """Generate text using the locally trained Transformer."""
    load_model()

    return _model.generate(
        start=prompt,
        id_to_char=_id_to_char,
        char_to_id=_char_to_id,
        max_new_tokens=max_new_tokens,
    )


def get_model_info():
    """Return information about the loaded checkpoint."""
    load_model()
    return _checkpoint_info.copy()
