from pathlib import Path

from app.models.document import Document

class DocumentLoader:

    def load(self, path: str) -> Document:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(path)

        content = file_path.read_text(encoding="utf-8")

        return Document(
            id=file_path.stem,
            content=content,
            metadata={
                "source": file_path.name,
                "file_type": file_path.suffix,
            }
        )