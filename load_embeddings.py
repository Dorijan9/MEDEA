import torch

# Load the embeddings
embeddings = torch.load('pinnacle_mg_embed.pth', map_location='cpu', weights_only=True)

print(f"Type:  {type(embeddings)}")
print(f"Shape: {embeddings.shape}")   # (218, 128)
print(f"Dtype: {embeddings.dtype}")
print(f"Min:   {embeddings.min():.4f}")
print(f"Max:   {embeddings.max():.4f}")
print(f"Mean:  {embeddings.mean():.4f}")
print(f"Std:   {embeddings.std():.4f}")

# Access individual embeddings
first_embedding = embeddings[0]    # shape (128,)
print(f"\nFirst embedding (first 10 values): {first_embedding[:10]}")

# Example: compute cosine similarity between first two embeddings
from torch.nn.functional import cosine_similarity
sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
print(f"\nCosine similarity between embedding 0 and 1: {sim.item():.4f}")
