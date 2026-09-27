import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

QUERY = "How many days can employees work remotely?"

DOCUMENTS = {
    "remote_work_policy": (
        "Employees may work remotely for up to three days per week. "
        "Employees must be available during core working hours from "
        "10:00 AM to 4:00 PM. Remote work must be approved by the "
        "employee's manager."
    ),
    "security_policy": (
        "Employees must use multi-factor authentication when accessing "
        "company systems. Customer data must only be stored in approved "
        "company systems. Employees must report suspected security "
        "incidents immediately."
    ),
    "travel_policy": (
        "International business travel requires approval from the "
        "employee's department head. Hotel expenses are reimbursable "
        "up to the company's approved accommodation limit."
    ),
}


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def dot_product(a, b):
    return np.dot(a, b)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def squared_euclidean_distance(a, b):
    return np.sum((a - b) ** 2)


def run_experiment(normalize_embeddings: bool):

    print("\n" + "=" * 90)
    print(f"NORMALIZE EMBEDDINGS = {normalize_embeddings}")
    print("=" * 90)

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        QUERY,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    document_embeddings = model.encode(
        list(DOCUMENTS.values()),
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    print("\nVector magnitudes:")

    print(
        f"Query: "
        f"{np.linalg.norm(query_embedding):.10f}"
    )

    for document_name, embedding in zip(
        DOCUMENTS.keys(),
        document_embeddings,
    ):
        print(
            f"{document_name}: "
            f"{np.linalg.norm(embedding):.10f}"
        )

    print("\nMetrics:")

    print(
        f"{'Document':<25}"
        f"{'Cosine':>12}"
        f"{'Dot':>12}"
        f"{'Euclidean':>15}"
        f"{'Squared L2':>15}"
    )

    print("-" * 90)

    for document_name, document_embedding in zip(
        DOCUMENTS.keys(),
        document_embeddings,
    ):

        cosine = cosine_similarity(
            query_embedding,
            document_embedding,
        )

        dot = dot_product(
            query_embedding,
            document_embedding,
        )

        euclidean = euclidean_distance(
            query_embedding,
            document_embedding,
        )

        squared_l2 = squared_euclidean_distance(
            query_embedding,
            document_embedding,
        )

        print(
            f"{document_name:<25}"
            f"{cosine:>12.6f}"
            f"{dot:>12.6f}"
            f"{euclidean:>15.6f}"
            f"{squared_l2:>15.6f}"
        )


def main():

    print(f"Query: {QUERY}")

    run_experiment(
        normalize_embeddings=True
    )

    run_experiment(
        normalize_embeddings=False
    )


if __name__ == "__main__":
    main()