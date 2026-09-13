from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("Loading the model...")
model = SentenceTransformer('all-MiniLM-L6-v2')     
print("Model loaded successfully.")

# Step 2: Define our input sentences
sentences = [
    "A dog is chasing a bright red ball on a sunny green field.",
    "The canine pursues a crimson sphere across the lawn.",
    "I'm planning to bake an apple pie for the picnic next weekend.",
    "A new laptop was purchased from the tech store today."
]

# Step 3: Generate embeddings
print("\nGenerating embeddings for the sentences...")
embeddings = model.encode(sentences)

# Look at the shape and size of the vectors
print(f"Shape of the embeddings tensor: {embeddings.shape}")
print(f"Embedding dimension (d_k): {embeddings.shape[1]}")
print(f"Sample of the first embedding vector: {embeddings[0][:5]}...")

similarity_matrix=cosine_similarity(embeddings)

S1_S2_sim=similarity_matrix[0,1]
S1_S3_sim=similarity_matrix[0,2]
S1_S4_sim=similarity_matrix[0,3]

print("\n--Semantic Similarity Analysis(Cosine Score:1.0=identical, 0.0=Unrelated/orthogonal)--")
print(f"Similarity between Sentence 1 and Sentence 2: {S1_S2_sim:.4f}")
print(f"Similarity between Sentence 1 and Sentence 3: {S1_S3_sim:.4f}")
print(f"Similarity between Sentence 1 and Sentence 4: {S1_S4_sim:.4f}")