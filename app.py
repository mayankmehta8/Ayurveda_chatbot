from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr
import os

# === Step 1: Load and Split PDFs ===
def load_documents(path="documents"):
    all_docs = []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# === Step 2: Embed and Store ===
def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    return db

# === Step 3: Load Vectorstore ===
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# === Step 4: Load Local LLM ===
def load_llm():
    model_id = "models/mistral-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

# === Step 5: Build RAG Chain ===
def build_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    llm = load_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa

# === Step 6: Gradio Interface ===
qa_chain = None

def setup():
    global qa_chain
    if not os.path.exists("faiss_index"):
        print("Embedding docs for the first time...")
        docs = load_documents()
        chunks = split_documents(docs)
        create_vectorstore(chunks)
    qa_chain = build_chain()

def chatbot(query):
    if not qa_chain:
        return "Chatbot not initialized. Please reload."
    result = qa_chain.run(query)
    return result

setup()

iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="📚 Local RAG Chatbot",
    description="Ask questions from your local PDFs using a fully offline RAG pipeline!",
)

iface.launch()
