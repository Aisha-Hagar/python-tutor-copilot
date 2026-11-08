import logging
import uuid
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src import config

logger = logging.getLogger(__name__)

def load_pdf(pdf_path: str) -> list[dict]:
    """
    Load PDF file and return list of pages, each with text and page number.
    """
    path = Path(pdf_path)
    if not  path.exists():
        raise FileNotFoundError(f"PDF file {path.resolve} does not exist")

    logger.info(f"Loading PDF file: {path}")
    pdf_reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i+1})
        else:
            logger.warning(f"Page {i+1} has no text.")

    logger.info(f"Loaded PDF file: {pdf_path}")
    logger.info(f"Extracted {len(pages)} pages with text.")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split each page's text into overlapping chunks using recursive
    character splitting. Keep page number as metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({"text": chunk_text,
                           "page": page["page"]
                           })

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    return chunks


def get_embeddings() -> OllamaEmbeddings:
    """
    Initialize the embedding model.
    """
    logger.info(f"Initializing embeddings model: {config.EMBEDDING_MODEL}")
    return OllamaEmbeddings(model=config.EMBEDDING_MODEL)


def get_qdrant_client() -> QdrantClient:
    """
    Create a QdrantClient.
    """
    logger.info(f"Connecting to Qdrant at {config.QDRANT_URL}")
    return QdrantClient(url=config.QDRANT_URL,
                        api_key=config.QDRANT_API_KEY,
                        timeout=120)

def reset_collection(client: QdrantClient) -> None:
    """
    Delete the collection of exists, then recreate it with the correct
    vector dimension.
    """
    collections = [c.name for c in client.get_collections().collections]
    if config.COLLECTION_NAME in collections:
        logger.info(f"Deleting existing collection: {config.COLLECTION_NAME}")
        client.delete_collection(collection_name=config.COLLECTION_NAME)

    logger.info(f"Recreating collection: {config.COLLECTION_NAME}"
                f"with vector size {config.VECTOR_DIM}"
                )
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=config.VECTOR_DIM, distance=Distance.COSINE)
    )

def index_chunks(client: QdrantClient, chunks: list[dict], embeddings: OllamaEmbeddings) -> None:
    """
    Embed each chunk and upsert into Qdrant in small batches.
    Uses deterministic IDs so re-runs overwrite rather than duplicate.
    """
    BATCH_SIZE = 100
    batch = []
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk["text"])
        if len(vector) != config.VECTOR_DIM:
            raise ValueError(
                f"Vector dimension mismatch: model produced {len(vector)}, "
                f"but collection expects {config.VECTOR_DIM}."
            )
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk-{i}"))
        batch.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": Path(config.PDF_PATH).name,
                    "chunk_id": i
                }
            )
        )

        if len(batch) >= BATCH_SIZE or i==len(chunks)-1:
            logger.info(f"Upserting batch of {len(batch)} points (chunk {i+1}/{len(chunks)}).")
            client.upsert(collection_name=config.COLLECTION_NAME, points=batch)
            batch = []

    logger.info("Upserted all batches.")


def run_ingestion():
    """
    Ingestion pipeline: load->chunk->embed->index.
    """
    pages = load_pdf(config.PDF_PATH)
    chunks = chunk_pages(pages)

    embeddings = get_embeddings()
    client = get_qdrant_client()

    reset_collection(client)
    index_chunks(client, chunks, embeddings)

    count = client.count(collection_name=config.COLLECTION_NAME).count
    logger.info(f"Collection contains {count} vectors.")