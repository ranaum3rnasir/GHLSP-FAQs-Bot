import os
import tempfile
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import base64

def decode_api_key(encoded_api_key):
    decoded_bytes = base64.b64decode(encoded_api_key.encode('utf-8'))
    decoded_str = str(decoded_bytes, 'utf-8')
    return decoded_str


# Set your encoded API key
api = "QUl6YVN5QXMyRUd1cjR4d3BrZW90cUtDakJPSmZLQkF2SVlZM3VN"
decoded_api_key = decode_api_key(api)

file_path = "PDF Data/Mineral Rights FAQ – For Land Investors.pdf"  # Example file path
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return splitter.split_text(text)


text = extract_text_from_pdf(file_path)
chunks = get_chunks(text)
docs = [Document(page_content=chunk, metadata={"source": "Mineral Rights FAQ – For Land Investors"}) for chunk in chunks]
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(decoded_api_key))
db = FAISS.from_documents(docs, embeddings)
db.save_local("Mineral_Rights_FAQ_embeddings")