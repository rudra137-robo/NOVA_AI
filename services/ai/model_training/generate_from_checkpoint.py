import torch
import torch.nn as nn
import torch.nn.functional as F

# Match the architecture used during training.
CONTEXT_SIZE = 64
EMBEDDING_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS = 2


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)),
        )

    def forward(self, x):
        _, sequence, _ = x.shape

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        scores = query @ key.transpose(-2, -1)
        scores = scores * (key.shape[-1] ** -0.5)

        scores = scores.masked_fill(
            self.mask[:sequence, :sequence] == 0,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)
        return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = EMBEDDING_SIZE // NUM_HEADS

        self.heads = nn.ModuleList([
            SelfAttentionHead(head_size)
            for _ in range(NUM_HEADS)
        ])

        self.projection = nn.Linear(
            EMBEDDING_SIZE,
            EMBEDDING_SIZE,
        )

    def forward(self, x):
        combined = torch.cat(
            [head(x) for head in self.heads],
            dim=-1,
        )
        return self.projection(combined)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, 4 * EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Linear(4 * EMBEDDING_SIZE, EMBEDDING_SIZE),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.norm1 = nn.LayerNorm(EMBEDDING_SIZE)
        self.norm2 = nn.LayerNorm(EMBEDDING_SIZE)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_SIZE,
        )
        self.position_embedding = nn.Embedding(
            CONTEXT_SIZE,
            EMBEDDING_SIZE,
        )
        self.blocks = nn.Sequential(*[
            TransformerBlock() for _ in range(NUM_LAYERS)
        ])
        self.final_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.output_head = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def forward(self, ids):
        _, sequence = ids.shape

        positions = torch.arange(sequence, device=ids.device)

        x = (
            self.token_embedding(ids)
            + self.position_embedding(positions)
        )

        x = self.blocks(x)
        x = self.final_norm(x)

        return self.output_head(x)

    @torch.no_grad()
    def generate(self, start, id_to_char, char_to_id, max_new_tokens=200):
        self.eval()

        start = "".join(ch for ch in start if ch in char_to_id)

        if not start:
            start = next(iter(char_to_id))

        ids = torch.tensor(
            [[char_to_id[ch] for ch in start]],
            dtype=torch.long,
        )

        for _ in range(max_new_tokens):
            context = ids[:, -CONTEXT_SIZE:]
            logits = self(context)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)

            next_id = torch.multinomial(probabilities, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

        return "".join(id_to_char[i] for i in ids[0].tolist())


def main():
    checkpoint_path = "model_training/tiny_transformer_best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    characters = checkpoint["characters"]
    char_to_id = {ch: i for i, ch in enumerate(characters)}
    id_to_char = {i: ch for ch, i in char_to_id.items()}

    model = TinyTransformer(len(characters))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", checkpoint_path)
    print("Best training step:", checkpoint["best_step"])
    print("Best validation loss:", checkpoint["best_validation_loss"])
    print("\nGenerated text:\n")
    print(model.generate("hello ", id_to_char, char_to_id))


if __name__ == "__main__":
    main()