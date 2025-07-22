# src/interface.py
import gradio as gr
from ingestao_pdf import extrair_texto_dos_pdfs
from vetorizacao import segmentar_documento, vetorizar_segmentos, criar_index_faiss, modelo_embedding
from sumarizacao import gerar_resumo
import numpy as np

# Carrega os documentos
DOCUMENTOS = extrair_texto_dos_pdfs("./data/pdfs")
NOMES_ARQUIVOS = list(DOCUMENTOS.keys())

def responder(pergunta, nome_arquivo):
    texto = DOCUMENTOS[nome_arquivo]
    segmentos = segmentar_documento(texto)
    vetores = vetorizar_segmentos(segmentos)
    index = criar_index_faiss(np.array(vetores))
    query_vec = modelo_embedding.encode([pergunta])
    D, I = index.search(np.array(query_vec), k=5)
    contexto = [segmentos[i] for i in I[0]]
    resposta = gerar_resumo(" ".join(contexto))
    return resposta

iface = gr.Interface(
    fn=responder,
    inputs=[
        gr.Textbox(label="Digite sua pergunta"),
        gr.Dropdown(choices=NOMES_ARQUIVOS, label="Selecione um material")
    ],
    outputs="text",
    title="Prometeia: Estudos para Concurso com IA",
    description="Ferramenta baseada em RAG para responder perguntas com base em materiais de estudo."
)

if __name__ == "__main__":
    iface.launch()
