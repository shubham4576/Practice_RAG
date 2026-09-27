from app.models.document import Document


class Chunker:

    def __init__(self, chunk_size: int = 300):
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> list[Document]:
        words = document.content.split()

        chunks = []

        for start in range(0, len(words), self.chunk_size):
            chunk_words = words[start:start + self.chunk_size]

            chunk_id = f"{document.id}-chunk-{len(chunks)}"

            chunks.append(
                Document(
                    id=chunk_id,
                    content=" ".join(chunk_words),
                    metadata={
                        **document.metadata,
                        "document_id": document.id,
                        "chunk_index": len(chunks),
                    }
                )
            )

        return chunks