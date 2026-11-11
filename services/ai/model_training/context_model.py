
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

text = """
hello world
hello nova
nova is learning
we are building nova
hello world
nova learns from text
"""

# Character vocabulary
characters = sorted(set(text))
vocab_size = len(characters)

char_to_id = {ch: i for i, ch in enumerate(characters)}
id_to_char = {i: ch for ch, i in char_to_id.items()}

data = torch.tensor(
    [char_to_id[ch] for ch in text],
    dtype=torch.long,
)

# Training settings
context_size = 8
embedding_size = 32

# Every input contains the previous eight characters.
# The target is the character immediately after them.
def make_examples(data):
    inputs = []
    targets = []

    for i in range(context_size, len(data)):
        inputs.append(data[i - context_size:i])
        targets.append(data[i])

    return torch.stack(inputs), torch.stack(targets)


x, y = make_examples(data)


class ContextModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.network = nn.Sequential(
            nn.Linear(context_size * embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, vocab_size),
        )

    def forward(self, ids):
        # [batch, context] -> [batch, context, embedding]
        embeddings = self.embedding(ids)

        # Combine all context embeddings into one vector.
        combined = embeddings.reshape(ids.shape[0], -1)

        # Predict the next character.
        return self.network(combined)


model = ContextModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(501):
    logits = model(x)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")


@torch.no_grad()
def generate(start="hello ", length=100):
    model.eval()

    start = "".join(ch for ch in start if ch in char_to_id)
    if not start:
        start = "h" if "h" in char_to_id else characters[0]

    result = start
    ids = [char_to_id[ch] for ch in start]

    for _ in range(length):
        context = ids[-context_size:]

        # Left-pad short contexts with spaces.
        context = [char_to_id[" "]] * (context_size - len(context)) + context
        context_tensor = torch.tensor([context])

        logits = model(context_tensor)
        probabilities = F.softmax(logits[0], dim=-1)
        next_id = torch.multinomial(probabilities, 1).item()

        ids.append(next_id)
        result += id_to_char[next_id]

    return result


print("\nGenerated text:")
print(generate())
