from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from pdf_summarizer import (
    chat_about_summaries,
    create_client,
    list_summary_files,
    summarize_folder,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_project_path(folder: str) -> Path:
    path = Path(folder).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


st.set_page_config(page_title="PDF Summarizer & Chat", page_icon="📚", layout="wide")
st.title("PDF Summarizer & Chat")
st.caption("Summarize a folder of PDFs, then ask questions grounded in the saved summaries.")

with st.sidebar:
    st.header("Groq model")
    st.caption("Configure a Groq API key and optional model ID in `.env`.")
    st.code("GROQ_API_KEY\nGROQ_MODEL_ID", language="text")

batch_column, chat_column = st.columns([1, 1], gap="large")

with batch_column:
    st.subheader("1. Summarize a folder")
    input_folder = st.text_input("Folder containing PDFs", value="input_pdfs")
    output_folder = st.text_input("Folder for summaries", value="summaries")
    st.caption("Each PDF produces a Markdown summary in the output folder.")

    if st.button("Run PDF summarizer", type="primary", use_container_width=True):
        source = _resolve_project_path(input_folder)
        destination = _resolve_project_path(output_folder)
        if not input_folder.strip():
            st.error("Enter the path to a folder containing PDF files.")
        elif not source.is_dir():
            st.error(f"Input folder does not exist: {source}")
        else:
            pdfs = sorted(
                path
                for path in source.iterdir()
                if path.is_file() and path.suffix.lower() == ".pdf"
            )
            if not pdfs:
                st.warning(f"No PDF files were found in {source}.")
            else:
                try:
                    client = create_client()
                    destination.mkdir(parents=True, exist_ok=True)
                    progress = st.progress(0, text="Preparing PDFs...")
                    successes, failures = [], []
                    for index, pdf_path in enumerate(pdfs, start=1):
                        def update_progress(fraction: float, message: str) -> None:
                            progress.progress(
                                (index - 1 + fraction) / len(pdfs),
                                text=f"{pdf_path.name} ({index}/{len(pdfs)}): {message}",
                            )

                        try:
                            output_path = summarize_folder(
                                client,
                                pdf_path,
                                destination,
                                progress_callback=update_progress,
                            )
                            successes.append(output_path.name)
                        except Exception as exc:
                            failures.append((pdf_path.name, str(exc)))
                    progress.progress(1.0, text="Batch complete.")
                    if successes:
                        st.success(
                            f"Created {len(successes)} summary file(s) in {destination}."
                        )
                        for name in successes:
                            st.write(f"- `{name}`")
                    for name, message in failures:
                        st.error(f"{name}: {message}")
                    if not successes:
                        st.warning("No summaries were created. See the errors above.")
                except Exception as exc:
                    st.error(f"Could not start summarization: {exc}")

with chat_column:
    st.subheader("2. Chat with the summaries")
    chat_folder = st.text_input(
        "Folder containing generated summaries",
        value=output_folder,
        key="chat_folder",
    )
    summary_files = list_summary_files(_resolve_project_path(chat_folder))
    if summary_files:
        st.caption(
            f"Ready to search {len(summary_files)} summary file(s). "
            "Answers include source filenames."
        )
    else:
        st.info("Run the summarizer first, or enter a folder that contains `.md` summaries.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the summarized PDFs...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching summaries and preparing an answer..."):
                try:
                    if not summary_files:
                        raise ValueError(
                            "No Markdown summary files were found in the selected folder."
                        )
                    answer = chat_about_summaries(
                        create_client(), question, summary_files
                    )
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as exc:
                    st.error(f"Could not answer the question: {exc}")

