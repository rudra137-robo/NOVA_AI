from pathlib import Path
import fitz


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md"
}


def load_document(file_path: Path) -> list[dict]:

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document format: {extension}"
        )

    pages = []

    # PDF processing
    if extension == ".pdf":

        with fitz.open(file_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "source": file_path.name,
                        "page": page_number,
                        "text": text
                    })

    # TXT and Markdown processing
    else:

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if text:
            pages.append({
                "source": file_path.name,
                "page": None,
                "text": text
            })

    return pages