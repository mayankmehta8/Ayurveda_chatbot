
# 🧠 Local RAG Chatbot with Gradio UI

A fully offline, private, and open-source Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your local PDF documents using:

- ✅ Local LLM (Mistral-7B-Instruct)
- ✅ Sentence Transformers for embeddings
- ✅ FAISS for vector search
- ✅ Gradio for UI
- ✅ LangChain for pipeline orchestration

---

## 🚀 Features

- 📄 Accepts and processes local PDFs
- 🤖 Uses Mistral-7B-Instruct as the language model (loaded locally)
- 🔍 Uses semantic search to retrieve relevant chunks
- 🧠 Combines retrieval + generation for accurate answers
- 💬 Provides a clean Gradio interface
- 📴 100% offline – no API keys, no data leaks

---

## 📁 Folder Structure

```
rag_bot/
├── app.py                  # Main chatbot script
├── documents/              # Place your PDFs here
├── models/                 # Contains downloaded LLM (Mistral)
└── faiss_index/            # Automatically created vector index
```

---

## 🧩 Requirements

Python 3.9+ and the following packages:

```bash
pip install langchain faiss-cpu gradio transformers accelerate sentence-transformers unstructured pdfminer.six
```

---

## 📥 Model Download

To download the model (Mistral-7B-Instruct) locally:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
save_path = "models/mistral-7b-instruct"

AutoTokenizer.from_pretrained(model_id).save_pretrained(save_path)
AutoModelForCausalLM.from_pretrained(model_id).save_pretrained(save_path)
```

> ✅ You only need to do this once. Make sure you are logged in to Hugging Face (`huggingface-cli login`).

---

## 🧠 How It Works

1. PDFs are loaded and split into chunks
2. Each chunk is converted into a vector using MiniLM embeddings
3. Vectors are stored in FAISS
4. At runtime, relevant chunks are retrieved based on your query
5. Mistral-7B-Instruct generates an answer using those chunks as context

---

## ▶️ Running the App

1. Place your PDFs inside the `documents/` folder
2. Run the chatbot:

```bash
python app.py
```

3. A Gradio interface will appear in your browser
4. Ask questions based on your PDF content!

---

## 🔐 Security Notice

FAISS index uses Python pickle for serialization. If you're loading an index you created yourself, it's safe to set:

```python
FAISS.load_local(..., allow_dangerous_deserialization=True)
```

Do **not** do this if loading untrusted or internet-downloaded indexes.

---

## 🧪 Future Improvements

- ✅ Add multi-PDF upload via Gradio
- 🧠 Add memory for multi-turn conversation
- 🗂️ Metadata-based filtering (e.g., search within document titles)
- ⚡ Use quantized LLMs for better performance on low-end machines

---

## 📜 License

MIT License – free for personal and commercial use.

---

## 🤝 Credits

- [Mistralai](https://huggingface.co/mistralai) for Mistral-7B-Instruct
- [LangChain](https://github.com/langchain-ai/langchain) for RAG pipelines
- [Hugging Face](https://huggingface.co/) for LLM & embedding models
- [Gradio](https://gradio.app) for UI
