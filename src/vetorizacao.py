# src/vetorizacao.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

modelo_embedding = SentenceTransformer("all-MiniLM-L6-v2")

def segmentar_documento(texto):
    paragrafos = texto.split("\n\n")
    segmentos = [p.strip() for p in paragrafos if len(p.strip()) > 50]
    return segmentos

def vetorizar_segmentos(segmentos):
    return modelo_embedding.encode(segmentos, convert_to_tensor=False)

def criar_index_faiss(vetores):
    dim = vetores.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vetores)
    return index
