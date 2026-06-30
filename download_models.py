from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading BGE-M3 embedding model...")
SentenceTransformer("BAAI/bge-m3")

print("Downloading BGE Reranker v2 M3...")
CrossEncoder("BAAI/bge-reranker-v2-m3")

print("Done.")
