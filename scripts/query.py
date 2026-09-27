from app.embeddings.embedder import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from experiments.vector_math import dot_product, euclidean_distance, cosine_similarity

import numpy as np

def main():

    queries = ["How many days can employees work remotely?"]

    for query in queries:
        print(f"Query: {query}")

        embedder = EmbeddingModel()
        vector_store = VectorStore()

        query_embedding = embedder.embed_query(query)

        print(f"Query dimensions: {len(query_embedding)}")
        print(
            f"Query magnitude: "
            f"{np.linalg.norm(query_embedding):.10f}"
        )

        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=3,
            filters={
                "country": "India"
            }
        )

        stored_embeddings = results["embeddings"][0]

        for i, document in enumerate(results["documents"][0]):

            document_embedding = stored_embeddings[i]

            cosine = cosine_similarity(
                query_embedding,
                document_embedding,
            )

            euclidean = euclidean_distance(
                query_embedding,
                document_embedding,
            )

            dot = dot_product(
                query_embedding,
                document_embedding,
            )

            chroma_distance = results["distances"][0][i]

            print("\n" + "=" * 70)

            print(f"Rank: {i + 1}")
            print(f"Content: {document}")

            print("\nOur calculations:")
            print(f"Cosine similarity : {cosine}")
            print(f"Euclidean distance: {euclidean}")
            print(f"Dot product       : {dot}")

            print("\nChroma:")
            print(f"Distance          : {chroma_distance}")


if __name__ == "__main__":
    main()