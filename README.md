# ChatDoc-AI

A full-stack Retrieval-Augmented Generation (RAG) chatbot for querying uploaded documents. Users can upload PDF, DOCX, or TXT files, index them with embeddings, and ask questions about the content through a chat interface.

The project combines a FastAPI backend for document ingestion and retrieval with a Streamlit frontend for interactive document Q&A.

## Features

- Upload multiple PDF, DOCX, and TXT files
- Extract text and split documents into chunks
- Generate embeddings with sentence-transformers
- Store and retrieve vectors using FAISS
- Ask natural-language questions grounded in uploaded documents
- Maintain short conversational memory during a session
- Works with Google Gemini for answer generation
- Simple local web UI for end-to-end usage

## Tech Stack

- Python
- FastAPI
- Streamlit
- FAISS
- LangChain
- Hugging Face Embeddings
- Google Generative AI (Gemini)
- pypdf, docx2txt, python-dotenv

## Project Structure

```text
Document-RAG-Chatbot/
├── backend/
│   ├── create_index.py
│   └── main.py
├── frontend/
│   └── app.py
├── faiss_index/
├── requirements.txt
├── .env
└── README.md
```

## How It Works

1. A user uploads one or more documents from the Streamlit app.
2. The backend accepts the files and saves them temporarily.
3. The document loader extracts text from PDFs, DOCX, or TXT files.
4. The text is chunked into smaller segments.
5. Embeddings are generated and stored in a FAISS vector index.
6. When a question is asked, the app retrieves the most relevant chunks.
7. The LLM answers using the document context and the recent chat history.

## Prerequisites

- Python 3.10+
- A Google API key with access to Gemini models
- pip package manager

## Setup

1. Clone the repository

```bash
git clone https://github.com/01-aniket-maity/Document-RAG-Chatbot.git
cd Document-RAG-Chatbot
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

## Run the Application

### Start the backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

### Start the frontend

In a new terminal:

```bash
cd frontend
streamlit run app.py
```

Open the local Streamlit URL in your browser (usually `http://localhost:8501`).

## API Endpoints

### `POST /upload`
Uploads one or more documents for indexing.

Supported formats:
- PDF
- DOCX
- TXT

### `POST /query`
Sends a question and gets an answer grounded in the indexed document content.

Example request:

```json
{
  "text": "What is the main topic of this document?"
}
```

### `GET /`
Returns a welcome message.

## Usage

1. Open the Streamlit app.
2. Upload your documents.
3. Click `Start Uploading`.
4. Ask questions in the chat area.
5. The chatbot will answer based on the indexed content.

## Notes

- The FAISS index is stored under the `faiss_index/` directory.
- This project uses temporary in-memory chat history on the server side.
- The app is designed for local development and lightweight document Q&A workflows.

## Future Improvements

- Add support for more document types
- Persist chat history in a database
- Add user authentication
- Improve retrieval quality with hybrid search
- Support multi-user document management
- Add deployment-ready Docker configuration

