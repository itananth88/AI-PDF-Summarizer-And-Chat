# PDF Summarizer & Chat

A local Streamlit application that summarizes PDFs in a folder into Markdown
files, then answers questions using those saved summaries. It uses chat models
available through Hugging Face Inference Providers.

## Pipeline

1. Install Python 3.10 or later.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Create a Hugging Face access token with permission to use Inference
   Providers. Copy `.env.example` to `.env`, then set `HF_TOKEN` and
   `HF_MODEL_ID` to your token and a chat-completion model available to your
   account. Keep `.env` private; never paste the token into chat or commit it.
5. Launch the interface:

   ```powershell
   streamlit run app.py
   ```

6. In the first panel, enter the input PDF folder and output folder, then
   select **Run PDF summarizer**. The app defaults to `input_pdfs` and
   `summaries`. Successful PDFs produce `<pdf-name>.summary.md` files.
   Individual failures are reported while the remaining PDFs continue
   processing.
7. In the chat panel, select the folder containing those summary files and ask
   a question. Answers cite the summary filenames used as evidence.

## Notes and limitations

- The app extracts selectable PDF text and falls back to local Tesseract OCR
  for pages without selectable text. Install Tesseract with English language
  data and make sure it is on `PATH`; OCR output can contain recognition errors.
- Large PDFs are summarized in overlapping text chunks before a final
  synthesis. The chat retrieves matching excerpts from generated summaries;
  answers can still be incomplete, so verify consequential details against
  the original PDF.
- Extracted PDF text and chat context are sent to the Hugging Face Inference
  Provider serving the selected model. Do not process sensitive documents
  unless your data-handling requirements allow this.
- This initial version processes PDFs in the selected folder (not nested
  subfolders) and writes Markdown summaries.
