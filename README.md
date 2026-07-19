# 📊 Financial AI Analyzer

A Retrieval-Augmented Generation (RAG) based Financial AI Analyzer built using **Python**, **LangChain**, **ChromaDB**, **Hugging Face Embeddings**, and **Ollama**. The application enables users to upload financial datasets, store them in a vector database, and ask financial questions in natural language with conversation memory for contextual follow-up queries.

---

## 🚀 Features

- 📁 Upload custom financial JSON files
- 📄 Automatic document ingestion and preprocessing
- ✂️ Intelligent document chunking
- 🧠 Semantic embeddings using Hugging Face
- 🗂️ ChromaDB vector database for efficient retrieval
- 🤖 AI-powered financial question answering using Ollama
- 💬 Conversation memory for follow-up questions
- 🔍 Semantic search over financial documents
- 🖥️ Interactive Command Line Interface (CLI)
- 🛠️ Debug mode to inspect retrieved chunks and metadata

---

## 🛠️ Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face Embeddings (`all-MiniLM-L6-v2`)
- Ollama (Llama 3.2)
- python-dotenv

---

## 📂 Project Structure

```
Finance_analyzer/
│
├── data/
│   ├── fmp/
│   ├── reports/
│   └── uploads/
│
├── db/
│   └── chroma/
│
├── ingestion.py
├── retrieval.py
├── main.py
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dikshit781/Financial-AI-Analyzer.git

cd Financial-AI-Analyzer
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download Ollama from:

https://ollama.com/download

Pull the model:

```bash
ollama pull llama3.2
```

Start Ollama:

```bash
ollama serve
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root.

```env
OLLAMA_MODEL=llama3.2

DATA_FOLDER=data
FMP_FOLDER=data/fmp
REPORTS_FOLDER=data/reports
UPLOAD_FOLDER=data/uploads

CHROMA_FOLDER=db/chroma

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

---

## ▶️ Running the Project

```bash
python main.py
```

---

## 📌 Menu

```
1. Search Existing Companies
2. Upload Your Own Financial Data
3. Exit
```

---

## 💬 Example Queries

### Company Information

```
Tell me about Apple.
```

```
Give an overview of Microsoft.
```

### Financial Questions

```
What is Tesla's revenue?
```

```
What is Amazon's operating income?
```

```
What is Alphabet's net income?
```

### Follow-up Questions

```
Tell me about Apple.

What is its revenue?

What about its operating profit?

Compare it with Microsoft.
```

---

## 🔄 RAG Pipeline

```
Financial Documents
        │
        ▼
Document Loader
        │
        ▼
Text Chunking
        │
        ▼
Embeddings
        │
        ▼
ChromaDB
        │
        ▼
Semantic Retrieval
        │
        ▼
Ollama (Llama 3.2)
        │
        ▼
Financial Answer
```

---

## 📁 Supported File Types

- JSON
- CSV *(upload support can be extended)*
- TXT
- PDF *(upload support can be extended)*
- XLSX *(upload support can be extended)*

---

## 📈 Future Improvements

- History-aware retrieval
- Financial ratio analysis
- Interactive charts and visualizations
- Hybrid Search (BM25 + Vector Search)
- Multi-file financial analysis
- Company comparison dashboard
- Web interface using Streamlit or React
- API deployment with FastAPI

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "feat: add new feature"
```

4. Push the branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Dikshit Dhiman**

GitHub: https://github.com/Dikshit781

LinkedIn: https://www.linkedin.com/in/dikshit-dhiman/

---
⭐ If you found this project useful, consider giving it a star!

