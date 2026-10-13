
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# RAG imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from create_index import build_index


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing from your .env file.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR.parent / "faiss_index"

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Temporary in-memory conversation history.
# It is lost when the FastAPI process restarts.
chat_history = []
MAX_HISTORY_MESSAGES = 20  # 10 user/assistant exchanges


# --------------------------------------------------
# FastAPI setup
# --------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Embeddings and Gemini
# --------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.4,
)

SYSTEM_PROMPT = """
You are a helpful document assistant.

Use the retrieved document context and chat history to answer the question.
Keep the answer to a maximum of three sentences.
If the answer cannot be found in the provided document context,
say that you don't know.

Document context:
{context}

Conversation history:
{chat_history}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


# --------------------------------------------------
# Initialize or load FAISS
# --------------------------------------------------

db = None
rag_chain = None


def refresh_rag_chain():
    """Load the saved FAISS index and create a retrieval chain."""
    global db, rag_chain

    index_file = FAISS_PATH / "index.faiss"
    metadata_file = FAISS_PATH / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        db = None
        rag_chain = None
        print("No FAISS index found. Upload documents first.")
        return

    print("Loading FAISS index...")

    db = FAISS.load_local(
        str(FAISS_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(
        retriever,
        qa_chain,
    )

    print("RAG chain is ready.")


# Load the existing index if available.
refresh_rag_chain()


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QueryRequest(BaseModel):
    text: str


# --------------------------------------------------
# Upload multiple documents
# --------------------------------------------------

@app.post("/upload")
def upload_files(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one file.",
        )

    saved_paths = []

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for uploaded_file in files:
                filename = Path(uploaded_file.filename or "").name
                extension = Path(filename).suffix.lower()

                if not filename or extension not in ALLOWED_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Unsupported file: {filename}. "
                            "Upload PDF, DOCX, or TXT files."
                        ),
                    )

                content = uploaded_file.file.read()

                if not content:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{filename} is empty.",
                    )

                # Use a unique path if two files have the same name.
                destination = temp_path / f"{len(saved_paths)}_{filename}"
                destination.write_bytes(content)
                saved_paths.append(destination)

            # Append these files to the existing FAISS index.
            build_index(saved_paths)

        # Make newly uploaded documents available to retrieval.
        refresh_rag_chain()

        return {
            "message": "Documents uploaded and indexed successfully.",
            "filenames": [
                Path(file.filename or "").name for file in files
            ],
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index uploaded documents: {str(exc)}",
        ) from exc

    finally:
        for uploaded_file in files:
            uploaded_file.file.close()


# --------------------------------------------------
# Ask questions with temporary conversation memory
# --------------------------------------------------

@app.post("/query")
def query_rag(req: QueryRequest):
    global chat_history

    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a document before asking questions.",
        )

    try:
        response = rag_chain.invoke({
            "input": req.text,
            "chat_history": list(chat_history),
        })

        answer = response.get("answer", "No answer found.")

        # Save this exchange in server memory.
        chat_history.extend([
            HumanMessage(content=req.text),
            AIMessage(content=answer),
        ])

        # Retain only the most recent 10 exchanges.
        chat_history = chat_history[-MAX_HISTORY_MESSAGES:]

        return {"answer": answer}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate an answer: {str(exc)}",
        ) from exc


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the document RAG chatbot. "
                   "Visit /docs to get started."
    }
