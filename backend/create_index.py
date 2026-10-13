#read files
#break in chunks
#embeddings
#store in vector db -> FAISS


from pathlib import Path
from typing import Union, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings


# Store the FAISS index in the project's faiss_index folder
BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR.parent / "faiss_index"


def load_uploaded_file(file_path: Path):
    """Load an uploaded PDF, DOCX, or TXT file."""

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))

    elif extension == ".txt":
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

    else:
        raise ValueError(
            f"Unsupported file type: {file_path.name}"
        )

    documents = loader.load()

    # Keep the filename as metadata
    for document in documents:
        document.metadata["source"] = file_path.name

    return documents


def build_index(
    file_paths: Union[str, Path, Iterable[Union[str, Path]]]
):
    """Create a FAISS index or append uploaded documents to it."""

    # Accept either one file or multiple files
    if isinstance(file_paths, (str, Path)):
        file_paths = [Path(file_paths)]
    else:
        file_paths = [Path(path) for path in file_paths]

    if not file_paths:
        raise ValueError("No files were provided.")

    # 1. Load all uploaded files
    documents = []

    for file_path in file_paths:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        documents.extend(load_uploaded_file(file_path))

    if not documents:
        raise ValueError("No readable content found.")

    # 2. Break documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were generated.")

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Load the existing index or create a new one
    index_file = FAISS_PATH / "index.faiss"
    metadata_file = FAISS_PATH / "index.pkl"

    if index_file.exists() and metadata_file.exists():

        db = FAISS.load_local(
            str(FAISS_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Append newly uploaded document chunks
        db.add_documents(chunks)

    else:
        db = FAISS.from_documents(
            chunks,
            embeddings
        )

    # 5. Save the updated index
    FAISS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    db.save_local(str(FAISS_PATH))

    print(
        f"Indexed {len(chunks)} chunks "
        f"from {len(file_paths)} uploaded file(s)."
    )

    return db



