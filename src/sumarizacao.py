# src/sumarizacao.py
from transformers import pipeline

sumarizador = pipeline("summarization", model="google/pegasus-xsum")

def gerar_resumo(texto):
    return sumarizador(texto[:1024])[0]['summary_text']
