
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# 1. Our tiny training dataset.
# Later, we'll replace this with a proper dataset.
text = """
hello world
hello nova
nova is learning
we are building nova
hello world
nova learns from text
"""

# 2. Build a character vocabulary.
characters = sorted(set(text))
vocab_size = len(characters)

char_to_id = {ch: i for i, ch in enumerate(characters)}
id_to_char = {i: ch for ch, i in char_to_id.items()}

data = torch.tensor(
    [char_to_id[ch] for ch in text],
    dtype=torch.long,
)

# Each input character predicts the following character.
x = data[:-1]
y = data[1:]


# 3. A tiny neural language model.
class CharModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 16)
        self.output = nn.Linear(16, vocab_size)

    def forward(self, ids):
        embeddings = self.embedding(ids)
        logits = self.output(embeddings)
        return logits


model = CharModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

# 4. Train the model to predict the next character.
for step in range(501):
    logits = model(x)

    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")


# 5. Generate new characters from a starting character.
@torch.no_grad()
def generate(start="h", length=100):
    model.eval()

    if start not in char_to_id:
        start = "h" if "h" in char_to_id else characters[0]

    current = torch.tensor([char_to_id[start]])
    result = start

    for _ in range(length):
        logits = model(current)
        probabilities = F.softmax(logits[-1], dim=-1)

        next_id = torch.multinomial(probabilities, 1).item()
        result += id_to_char[next_id]
        current = torch.tensor([next_id])

    return result


print("\nGenerated text:")
print(generate())
