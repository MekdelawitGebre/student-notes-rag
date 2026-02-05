def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + size]
        chunks.append(" ".join(chunk))
    return chunks
