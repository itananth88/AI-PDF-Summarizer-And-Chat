import os
import re
import shutil
from pathlib import Path

import pymupdf
from huggingface_hub import InferenceClient

CHUNK_SIZE = 12_000
CHUNK_OVERLAP = 500
CHAT_CHUNK_SIZE = 3_000
MAX_CHAT_CONTEXTS = 5
WORD_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'-]{1,}")
STOP_WORDS = {
    "about", "after", "also", "and", "are", "because", "been", "before",
    "being", "between", "could", "does", "from", "have", "into", "more",
    "most", "only", "other", "should", "some", "than", "that", "their",
    "them", "there", "these", "they", "this", "those", "through", "what",
    "when", "where", "which", "while", "with", "would", "your",
}


def create_client() -> InferenceClient:
    token = os.getenv("HF_TOKEN", "").strip()
    model_id = os.getenv("HF_MODEL_ID", "").strip()
    if not token or not model_id:
        raise ValueError(
            "Configure HF_TOKEN and HF_MODEL_ID in the .env file."
        )
    return InferenceClient(model=model_id, token=token)


def _complete(client: InferenceClient, system_prompt: str, user_prompt: str) -> str:
    response = client.chat_completion(
        temperature=0.2,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The configured AI model returned an empty response.")
    return content.strip()


def _extract_pdf_text(pdf_path: Path) -> str:
    tesseract = shutil.which("tesseract")
    if not tesseract and os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        tesseract_dir = program_files / "Tesseract-OCR"
        executable = tesseract_dir / "tesseract.exe"
        if executable.is_file():
            os.environ["PATH"] = os.pathsep.join(
                (str(tesseract_dir), os.environ.get("PATH", ""))
            )
            tesseract = str(executable)

    if tesseract:
        tessdata_dir = Path(tesseract).resolve().parent / "tessdata"
        if (tessdata_dir / "eng.traineddata").is_file():
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)

    try:
        with pymupdf.open(pdf_path) as document:
            pages = []
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()
                if not page_text:
                    try:
                        page_text = page.get_textpage_ocr(
                            language="eng", dpi=300, full=True
                        ).extractText().strip()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Could not OCR page {page_number}. Install Tesseract "
                            "OCR with English language data and make sure it is on PATH. "
                            f"Tesseract error: {exc}"
                        ) from exc
                pages.append(page_text)
    except Exception as exc:
        raise RuntimeError(f"Could not read PDF: {exc}") from exc
    text = "\n\n".join(page for page in pages if page)
    if not text:
        raise ValueError(
            "No selectable text was found. Scanned/image-only PDFs need OCR before summarization."
        )
    return text


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


def summarize_folder(
    client: InferenceClient, pdf_path: Path, output_folder: Path
) -> Path:
    text = _extract_pdf_text(pdf_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    chunks = _split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    partial_summaries = []
    for index, chunk in enumerate(chunks, start=1):
        partial_summaries.append(
            _complete(
                client,
                "You are a careful document analyst. Summarize only the supplied text. "
                "Preserve key claims, evidence, names, dates, and uncertainty. "
                "Do not invent details.",
                f"Summarize this part of {pdf_path.name} (part {index} of {len(chunks)}):\n\n{chunk}",
            )
        )
    final_summary = _complete(
        client,
        "You create accurate, useful document summaries. Use only the supplied "
        "partial summaries; do not add facts. Distinguish explicit facts from "
        "conclusions and note important limitations.",
        "Create a structured summary with sections for Overview, Key points, "
        "Important details, and Limitations or open questions. Preserve material "
        "facts and avoid unsupported claims.\n\n"
        + "\n\n".join(
            f"Part {index} summary:\n{summary}"
            for index, summary in enumerate(partial_summaries, start=1)
        ),
    )
    output_path = output_folder / f"{pdf_path.stem}.summary.md"
    output_path.write_text(
        f"# Summary: {pdf_path.name}\n\n"
        f"> Generated from extracted PDF text; OCR may contain recognition errors. "
        f"Verify important details against the source PDF.\n\n"
        f"{final_summary}\n",
        encoding="utf-8",
    )
    return output_path


def list_summary_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.summary.md"))


def _terms(text: str) -> set[str]:
    return {
        word.lower()
        for word in WORD_PATTERN.findall(text)
        if word.lower() not in STOP_WORDS
    }


def _retrieve(question: str, files: list[Path]) -> list[tuple[str, str]]:
    query_terms = _terms(question)
    ranked = []
    for path in files:
        content = path.read_text(encoding="utf-8")
        for chunk in _split_text(content, CHAT_CHUNK_SIZE, 300):
            chunk_terms = _terms(chunk)
            overlap = query_terms & chunk_terms
            if overlap:
                # Prefer chunks matching several distinct query terms.
                score = sum(1 + chunk.lower().count(term) for term in overlap)
                ranked.append((score, path.name, chunk))
    ranked.sort(key=lambda result: result[0], reverse=True)
    return [(name, chunk) for _, name, chunk in ranked[:MAX_CHAT_CONTEXTS]]


def chat_about_summaries(
    client: InferenceClient, question: str, summary_files: list[Path]
) -> str:
    contexts = _retrieve(question, summary_files)
    if not contexts:
        return (
            "I couldn't find relevant information for that question in the available "
            "summaries. Try using terms that appear in the documents."
        )
    supplied_context = "\n\n".join(
        f"Source file: {name}\nRelevant summary text:\n{chunk}"
        for name, chunk in contexts
    )
    return _complete(
        client,
        "Answer using only the supplied summary excerpts. If they do not contain "
        "enough information, say so plainly. Do not infer missing facts. Cite each "
        "supporting source using its exact filename in square brackets, such as "
        "[report.summary.md].",
        f"Question: {question}\n\nSummary excerpts:\n{supplied_context}",
    )

