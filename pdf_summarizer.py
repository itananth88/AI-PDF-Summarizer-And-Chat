import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import pymupdf
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

CHUNK_SIZE = 12_000
CHUNK_OVERLAP = 500
CHAT_CHUNK_SIZE = 3_000
MAX_CHAT_CONTEXTS = 5
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
MAX_RATE_LIMIT_RETRIES = 3
MAX_RATE_LIMIT_WAIT_SECONDS = 65
RATE_LIMIT_DELAY_PATTERN = re.compile(r"try again in\s+([\d.]+)\s*s", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'-]{1,}")
STOP_WORDS = {
    "about", "after", "also", "and", "are", "because", "been", "before",
    "being", "between", "could", "does", "from", "have", "into", "more",
    "most", "only", "other", "should", "some", "than", "that", "their",
    "them", "there", "these", "they", "this", "those", "through", "what",
    "when", "where", "which", "while", "with", "would", "your",
}


def create_client() -> InferenceClient:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model_id = os.getenv("GROQ_MODEL_ID", DEFAULT_GROQ_MODEL).strip()
    if not api_key:
        raise ValueError(
            "Configure GROQ_API_KEY in the .env file. Create a key at "
            "https://console.groq.com/keys."
        )
    if not model_id:
        raise ValueError("GROQ_MODEL_ID cannot be empty.")
    return InferenceClient(base_url=GROQ_BASE_URL, api_key=api_key)


def _rate_limit_delay(error: HfHubHTTPError, attempt: int) -> float:
    headers = getattr(error.response, "headers", {})
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    reset_delay = headers.get("x-ratelimit-reset-tokens")
    if reset_delay:
        match = re.fullmatch(r"\s*([\d.]+)\s*s?\s*", reset_delay)
        if match:
            return float(match.group(1))

    match = RATE_LIMIT_DELAY_PATTERN.search(str(error))
    if match:
        return float(match.group(1))
    return float(2**attempt)


def _complete(
    client: InferenceClient,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    model_id = os.getenv("GROQ_MODEL_ID", DEFAULT_GROQ_MODEL).strip()
    request_options = {
        "model": model_id,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if model_id.startswith("openai/gpt-oss-"):
        request_options["extra_body"] = {
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
        }

    waited_seconds = 0.0
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            response = client.chat_completion(**request_options)
            break
        except HfHubHTTPError as exc:
            if (
                exc.response.status_code != 429
                or attempt >= MAX_RATE_LIMIT_RETRIES
            ):
                raise
            delay = _rate_limit_delay(exc, attempt)
            if waited_seconds + delay > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise
            time.sleep(delay)
            waited_seconds += delay

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
    client: InferenceClient,
    pdf_path: Path,
    output_folder: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    if progress_callback:
        progress_callback(0.0, "Extracting PDF text...")
    text = _extract_pdf_text(pdf_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    chunks = _split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    total_steps = len(chunks) + 1
    partial_summaries = []
    for index, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(
                (index - 1) / total_steps,
                f"Summarizing part {index} of {len(chunks)}...",
            )
        partial_summaries.append(
            _complete(
                client,
                "You are a careful document analyst. Summarize only the supplied text. "
                "Preserve key claims, evidence, names, dates, and uncertainty. "
                "Do not invent details.",
                f"Summarize this part of {pdf_path.name} (part {index} of {len(chunks)}):\n\n{chunk}",
            )
        )
        if progress_callback:
            progress_callback(
                index / total_steps,
                f"Completed part {index} of {len(chunks)}.",
            )
    if progress_callback:
        progress_callback(
            len(chunks) / total_steps,
            "Creating the final summary...",
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
        max_tokens=2048,
    )
    output_path = output_folder / f"{pdf_path.stem}.summary.md"
    output_path.write_text(
        f"# Summary: {pdf_path.name}\n\n"
        f"> Generated from extracted PDF text; OCR may contain recognition errors. "
        f"Verify important details against the source PDF.\n\n"
        f"{final_summary}\n",
        encoding="utf-8",
    )
    if progress_callback:
        progress_callback(1.0, "Summary saved.")
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

