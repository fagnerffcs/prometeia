# src/ingestao_pdf.py
import fitz  # PyMuPDF
import os

def extrair_texto_dos_pdfs(pasta_pdf):
    documentos = {}
    for arquivo in os.listdir(pasta_pdf):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta_pdf, arquivo)
            doc = fitz.open(caminho)
            texto = "\n".join([pagina.get_text() for pagina in doc])
            documentos[arquivo] = texto
    return documentos
