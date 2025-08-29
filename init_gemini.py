import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymilvus import connections

# Load .env environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")

if not ZILLIZ_CLOUD_URI or not ZILLIZ_CLOUD_TOKEN:
    raise ValueError("❌ ZILLIZ_CLOUD_URI or ZILLIZ_CLOUD_API_KEY not found in .env")

# Step 1: Load and parse the PDF
def load_pdf_documents(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# Step 2: Process and store embeddings
def process_and_store(file_path, collection_name="foxlearner_ch6_embeddings"):
    print(f"📄 Reading and splitting: {file_path}")
    docs = load_pdf_documents(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    vector_store = Milvus.from_documents(
        documents=split_docs,
        collection_name=collection_name,
        embedding=embeddings,
        connection_args={
            "uri": ZILLIZ_CLOUD_URI,
            "token": ZILLIZ_CLOUD_TOKEN,
            "secure": True
        }
    )

    print(f"✅ Embedding complete. Stored in Milvus collection: {collection_name}")
    return vector_store

# Step 3: Disconnect from Milvus safely
def safe_disconnect():
    if connections.has_connection("default"):
        connections.disconnect("default")
        print("🔌 Disconnected from Milvus.")

# === Main Execution ===
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(current_dir, "lecs105.pdf")

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"❌ PDF file not found at: {pdf_path}")

    try:
        retriever = process_and_store(pdf_path)
    finally:
        safe_disconnect()
