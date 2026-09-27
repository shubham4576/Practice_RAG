import chromadb

from app.models.document import Document


class VectorStore:

    def __init__(
            self,
            collection_name: str = "enterprise_documents",
            persist_directory: str = "./storage/chroma",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
        )

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        self.collection.add(
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            embeddings=embeddings,
            metadatas=[doc.metadata for doc in documents],
        )

    def search(
            self,
            query_embedding: list[float],
            top_k: int = 5,
            filters: dict | None = None,
    ):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
                "embeddings",
            ],
            where=filters,
        )