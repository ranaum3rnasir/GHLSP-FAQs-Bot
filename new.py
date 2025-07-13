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
    print("Received query request")
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    print("Received query request")
    user_query = request.form.get("query")
    print("Received query request")

    if not user_query:
        return jsonify({"error": "Missing query"}), 400


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
