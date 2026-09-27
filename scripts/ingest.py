from pathlib import Path

from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import Chunker
from app.embeddings.embedder import EmbeddingModel
from app.retrieval.vector_store import VectorStore


def get_department(file_path):

    mapping = {
        "remote_work_policy.txt": "HR",
        "remote_work_policy_uk.txt": "HR",
        "remote_work_policy_us.txt": "HR",
        "travel_policy.txt": "Finance",
        "security_policy.txt": "Security",
    }

    return mapping[file_path.name]

def get_country(file_path):

    mapping = {
        "remote_work_policy.txt": "India",
        "remote_work_policy_uk.txt": "UK",
        "remote_work_policy_us.txt": "US",
        "travel_policy.txt": "India",
        "security_policy.txt": "India",
    }

    return mapping[file_path.name]

def main():
    loader = DocumentLoader()
    chunker = Chunker(chunk_size=100)
    embedder = EmbeddingModel()
    vector_store = VectorStore()

    documents = []

    for file_path in Path("data/raw").rglob("*.txt"):
        print(f"Loading {file_path}")

        document = loader.load(str(file_path))

        document.metadata.update(
            {
                "department": get_department(file_path),
                "country": get_country(file_path),
                "status": "active",
                "version": "1.0",
            }
        )

        chunks = chunker.chunk(document)

        documents.extend(chunks)

    print(f"Created {len(documents)} chunks")

    texts = [doc.content for doc in documents]
    embeddings = embedder.embed(texts)

    print(
        f"Generated embeddings with dimension: "
        f"{len(embeddings[0])}"
    )

    vector_store.add(
        embeddings=embeddings,
        documents=documents,
    )

    print("Documents stored successfully")


if __name__ == "__main__":
    main()
