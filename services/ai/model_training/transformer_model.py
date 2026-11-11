
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducible initialization
torch.manual_seed(42)

# --------------------------------------------------
# 1. TRAINING DATA
# --------------------------------------------------

corpus = [
    "hello world",
    "hello nova",
    "nova is learning",
    "we are building nova",
    "nova learns from text",
    "artificial intelligence learns patterns",
    "we build useful systems",
    "machines learn from examples",
    "models predict the next character",
    "data helps us train models",
    "computers process information",
    "software follows instructions",
    "programmers solve problems",
    "python is a programming language",
    "neural networks learn from data",
    "training adjusts model parameters",
    "attention connects information across positions",
    "transformers process sequences of tokens",
    "a model predicts what comes next",
    "experiments help us understand learning",
    "good data helps models generalize",
    "validation measures performance on unseen examples",
    "overfitting happens when a model memorizes",
    "we test our model before deployment",
    "nova uses tools to perform tasks",
    "an assistant can answer questions",
    "algorithms transform input into output",
    "debugging helps us find software errors",
    "learning requires useful examples",
    "we improve systems through experiments",
]

# Hold out the final five sentences.
train_lines = corpus[:-5]
validation_lines = corpus[-5:]

# Give the training set more exposure without mixing in validation lines.
train_text = ("\n".join(train_lines) + "\n") * 20
validation_text = ("\n".join(validation_lines) + "\n") * 5

# Shared vocabulary: validation characters are known,
# but validation sentences are never used for training.
all_text = "\n".join(corpus) + "\n"
characters = sorted(set(all_text))

char_to_id = {ch: i for i, ch in enumerate(characters)}
id_to_char = {i: ch for ch, i in char_to_id.items()}
vocab_size = len(characters)

train_data = torch.tensor(
    [char_to_id[ch] for ch in train_text],
    dtype=torch.long,
)

validation_data = torch.tensor(
    [char_to_id[ch] for ch in validation_text],
    dtype=torch.long,
)

print("Training characters:", len(train_data))
print("Validation characters:", len(validation_data))

# Keep the model small enough for CPU training.
context_size = 64
batch_size = 16
embedding_size = 64
num_heads = 4
num_layers = 2
learning_rate = 0.001
training_steps = 1000

# --------------------------------------------------
# 2. SAMPLE TRAINING BATCHES
# --------------------------------------------------

def get_batch(source):
    starts = torch.randint(
        0,
        len(source) - context_size - 1,
        (batch_size,),
    )

    x = torch.stack([
        source[i:i + context_size]
        for i in starts
    ])

    y = torch.stack([
        source[i + 1:i + context_size + 1]
        for i in starts
    ])

    return x, y


# --------------------------------------------------
# 3. CAUSAL SELF-ATTENTION
# --------------------------------------------------

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

        # Position i may only attend to positions <= i.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_size, context_size)),
        )

    def forward(self, x):
        batch, sequence, channels = x.shape

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # Attention scores measure query-key similarity.
        scores = query @ key.transpose(-2, -1)
        scores = scores * (key.shape[-1] ** -0.5)

        # Block access to future characters.
        scores = scores.masked_fill(
            self.mask[:sequence, :sequence] == 0,
            float("-inf"),
        )

        weights = F.softmax(scores, dim=-1)

        # Combine information from earlier positions.
        return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = embedding_size // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        combined = torch.cat(
            [head(x) for head in self.heads],
            dim=-1,
        )

        return self.projection(combined)


# --------------------------------------------------
# 4. TRANSFORMER BLOCK
# --------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = MultiHeadAttention()
        self.feed_forward = FeedForward()

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        # Residual connections help information and gradients flow.
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


# --------------------------------------------------
# 5. LANGUAGE MODEL
# --------------------------------------------------

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_size
        )

        self.position_embedding = nn.Embedding(
            context_size, embedding_size
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock()
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_size)
        self.output_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, ids, targets=None):
        batch, sequence = ids.shape

        token_vectors = self.token_embedding(ids)

        positions = torch.arange(sequence)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, start="h", max_new_tokens=150):
        self.eval()

        start = "".join(ch for ch in start if ch in char_to_id)
        if not start:
            start = "h" if "h" in char_to_id else characters[0]

        ids = torch.tensor([
            [char_to_id[ch] for ch in start]
        ])

        for _ in range(max_new_tokens):
            context = ids[:, -context_size:]

            logits, _ = self(context)
            next_logits = logits[:, -1, :]

            probabilities = F.softmax(next_logits, dim=-1)

            next_id = torch.multinomial(
                probabilities, num_samples=1
            )

            ids = torch.cat([ids, next_id], dim=1)

        return "".join(
            id_to_char[i] for i in ids[0].tolist()
        )



# --------------------------------------------------
# 6. TRAIN
# --------------------------------------------------


model = TinyTransformer()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)

print("Vocabulary size:", vocab_size)
print("Model parameters:", sum(p.numel() for p in model.parameters()))
print("Training on CPU...")

best_validation_loss = float("inf")
best_step = 0
patience = 3
checks_without_improvement = 0

model.train()

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for name, source in [
        ("train", train_data),
        ("validation", validation_data),
    ]:
        losses = []

        for _ in range(10):
            x, y = get_batch(source)
            _, loss = model(x, y)
            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()
    return results
for step in range(training_steps + 1):
    x, y = get_batch(train_data)

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        losses = estimate_loss()

        train_loss = losses["train"]
        validation_loss = losses["validation"]

        print(
            f"Step {step:4d} | "
            f"Train: {train_loss:.4f} | "
            f"Validation: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_step = step
            checks_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "characters": characters,
                    "context_size": context_size,
                    "embedding_size": embedding_size,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "best_validation_loss": best_validation_loss,
                    "best_step": best_step,
                },
                "model_training/tiny_transformer_best.pt",
            )

            print("  Saved new best checkpoint.")
        else:
            checks_without_improvement += 1

        if checks_without_improvement >= patience:
            print("Early stopping: validation loss stopped improving.")
            break

print(f"\nBest step: {best_step}")
print(f"Best validation loss: {best_validation_loss:.4f}")

print("\nGenerated text:")
print(model.generate(start="hello ", max_new_tokens=200))

# Save our trained weights for later experiments.
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "characters": characters,
        "context_size": context_size,
        "embedding_size": embedding_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
    },
    "model_training/tiny_transformer.pt",
)

print("\nSaved trained model to model_training/tiny_transformer.pt")
