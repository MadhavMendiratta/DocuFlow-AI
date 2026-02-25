import logging
from .models import Document

logger = logging.getLogger(__name__)

# ── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_document(document: Document) -> str:
    """Route to the correct extractor based on file type."""
    file_path = document.file.path
    file_type = document.file_type.lower()

    extractors = {
        'pdf': extract_text_from_pdf,
        'txt': extract_text_from_txt,
        'docx': extract_text_from_docx,
    }

    extractor = extractors.get(file_type)
    if not extractor:
        raise ValueError(f"Unsupported file type: {file_type}")

    return extractor(file_path)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber with PyPDF2 fallback."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {i} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"pdfplumber: page {i} error: {e}")

        if text_parts:
            return '\n\n'.join(text_parts)
    except Exception as e:
        logger.warning(f"pdfplumber failed, falling back to PyPDF2: {e}")

    # Fallback to PyPDF2
    import PyPDF2

    text_parts = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i} ---\n{page_text}")
            except Exception as e:
                logger.warning(f"PyPDF2: page {i} error: {e}")

    return '\n\n'.join(text_parts)


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain-text file (tries multiple encodings)."""
    for encoding in ('utf-8', 'latin-1', 'cp1252'):
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode text file with any known encoding")


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a .docx file."""
    from docx import Document as DocxDocument

    doc = DocxDocument(file_path)
    return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())