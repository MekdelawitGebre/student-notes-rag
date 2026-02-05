import fitz

def load_pdf(path: str) -> list[dict]:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages
