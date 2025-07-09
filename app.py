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

# Decode API key
# Decode the Base64 API key
def decode_api_key(encoded_api_key):
    decoded_bytes = base64.b64decode(encoded_api_key.encode('utf-8'))
    decoded_str = str(decoded_bytes, 'utf-8')
    return decoded_str


# Set your encoded API key
api = "QUl6YVN5QXMyRUd1cjR4d3BrZW90cUtDakJPSmZLQkF2SVlZM3VN"
decoded_api_key = decode_api_key(api)
genai.configure(api_key=decoded_api_key) # type: ignore

app = Flask(__name__)

UPLOAD_DIR = "pdfs_uploaded"
EMBEDDING_DIR = "embeddings_pdf"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return splitter.split_text(text)

@app.route("/query", methods=["POST"])
def query():
    file = request.files.get("file")
    user_query = request.form.get("query")

    if not file or not user_query:
        return jsonify({"error": "Missing file or query"}), 400

    file_path = os.path.join(UPLOAD_DIR, file.filename) # type: ignore
    if not os.path.exists(file_path):
        file.save(file_path)
        text = extract_text_from_pdf(file_path)
        chunks = get_chunks(text)
        docs = [Document(page_content=chunk, metadata={"source": file.filename}) for chunk in chunks]
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(decoded_api_key))
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(decoded_api_key))
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(EMBEDDING_DIR)
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(decoded_api_key))
        db = FAISS.load_local(EMBEDDING_DIR, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever()
    matched_docs = retriever.get_relevant_documents(user_query, k=4)
    context = "\n\n".join([doc.page_content for doc in matched_docs])

    prompt = f"""
You are a helpful assistant. Use the provided document context to answer the user's question precisely and concisely.

Context:
{context}

Question: {user_query}

Answer:
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-002") # type: ignore
        response = model.generate_content(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"❌ LLM Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
