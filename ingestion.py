import json
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

from dotenv import load_dotenv
load_dotenv()

# Main folders
DATA_FOLDER = Path(os.getenv("DATA_FOLDER"))

FMP_FOLDER = Path(os.getenv("FMP_FOLDER"))

REPORTS_FOLDER = Path(os.getenv("REPORTS_FOLDER"))

CHROMA_FOLDER = Path(os.getenv("CHROMA_FOLDER"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

SUPPORTED_EXTENSIONS = {
    ".json",
    ".pdf",
    ".csv",
    ".xlsx",
    ".xls",
    ".txt"
}

# --------------------------------------------------
# find_data_files
# --------------------------------------------------

def find_data_files(data_folder=DATA_FOLDER):
    """
    Find all supported files inside data/fmp and data/reports.

    Args:
        data_folder:
            Main folder that contains fmp and reports.

    Returns:
        A sorted list of supported file paths.
    """

    data_folder = Path(data_folder)

    if not data_folder.exists():
        print(f"Error: Folder '{data_folder}' does not exist.")
        return []

    files = []

    for file_path in data_folder.rglob("*"):

        if file_path.is_file():
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(file_path)

    return sorted(files)



# --------------------------------------------------
# load_json_file
# --------------------------------------------------

def load_json_file(file_path):
    """
    Load and return data from one JSON file.

    Args:
        file_path:
            Path of the JSON file.

    Returns:
        Python dictionary or list if successful.
        None if the file cannot be loaded.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return None

    if file_path.suffix.lower() != ".json":
        print(f"Error: '{file_path}' is not a JSON file.")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data

    except json.JSONDecodeError:
        print(f"Error: '{file_path}' contains invalid JSON.")
        return None

    except Exception as error:
        print(f"Error while reading '{file_path}': {error}")
        return None
    

# --------------------------------------------------
# Convertr json into text.
# --------------------------------------------------

def make_readable_key(key):
    """
    Convert JSON keys into readable labels.

    Examples:
        marketCap -> Market Cap
        companyName -> Company Name
        full_time_employees -> Full Time Employees
    """

    key = key.replace("_", " ")

    key = re.sub(
        r"(?<!^)(?=[A-Z])",
        " ",
        key
    )

    return key.title()

def json_to_text(json_data):
    """
    Convert JSON data into readable text.

    Args:
        json_data:
            Dictionary or list loaded from a JSON file.

    Returns:
        A formatted text string.
    """

    lines = []

    if isinstance(json_data, list):

        for record in json_data:

            if not isinstance(record, dict):
                continue

            for key, value in record.items():

                readable_key = make_readable_key(key)

                lines.append(f"{readable_key} : {value}")

            lines.append("")

    elif isinstance(json_data, dict):

        for key, value in json_data.items():

            readable_key = make_readable_key(key)

            lines.append(f"{readable_key} : {value}")

    return "\n".join(lines)



# --------------------------------------------------
# Convert JSON files into a LangChain Documents
# --------------------------------------------------

def load_json_documents(data_folder=DATA_FOLDER):
    """
    Load all JSON files and convert them into LangChain Documents.
    """

    documents = []

    files = find_data_files(data_folder)

    for file_path in files:

        if file_path.suffix.lower() != ".json":
            continue

        json_data = load_json_file(file_path)

        if json_data is None:
            continue

        company = file_path.parent.name

        document_type = file_path.stem.replace("_", " ").title()

        json_text = json_to_text(json_data)

        if not json_text.strip():
            continue

        text = (
            f"Company : {company}\n"
            f"Document Type : {document_type}\n"
            f"{json_text}"
        )

        metadata = {
            "company": company,
            "document_type": file_path.stem,
            "source": str(file_path),
            "file_name": file_path.name
        }

        document = Document(
            page_content=text,
            metadata=metadata
        )

        documents.append(document)

    return documents

# --------------------------------------------------
# Split Documents
# --------------------------------------------------

def split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):
    """
    Split LangChain documents into smaller chunks.

    Args:
        documents:
            List of LangChain Document objects.

        chunk_size:
            Maximum size of each chunk.

        chunk_overlap:
            Overlap between consecutive chunks.

    Returns:
        List of chunked LangChain Documents.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    return chunks

#---------------------------------------------------
# Create the Embedding Model
# --------------------------------------------------

def create_embedding_model():
    """
    Create and return the HuggingFace embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embedding_model

# --------------------------------------------------
# Store Chunks in ChromaDB
# --------------------------------------------------

def build_vector_database():
    """
    Build and persist the Chroma vector database.
    """

    print("Loading JSON documents...")
    documents = load_json_documents()

    print(f"Documents loaded: {len(documents)}")

    print("\nSplitting documents...")
    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    print("\nLoading embedding model...")
    embedding_model = create_embedding_model()

    print("\nCreating Chroma database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_FOLDER)
    )

    print("\nVector database created successfully!")
    print(f"Total chunks stored: {len(chunks)}")

    return vectorstore

# --------------------------------------------------
# Test
# --------------------------------------------------
if __name__ == "__main__":

    build_vector_database()