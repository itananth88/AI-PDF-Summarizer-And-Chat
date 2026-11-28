import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pdf_summarizer
from huggingface_hub.errors import HfHubHTTPError


class GroqClientTests(unittest.TestCase):
    @patch("pdf_summarizer.InferenceClient")
    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True)
    def test_create_client_uses_groq_configuration(self, inference_client):
        pdf_summarizer.create_client()

        inference_client.assert_called_once_with(
            base_url=pdf_summarizer.GROQ_BASE_URL,
            api_key="test-key",
        )

    @patch.dict(
        os.environ,
        {"GROQ_MODEL_ID": "openai/gpt-oss-20b"},
        clear=True,
    )
    def test_completion_passes_model_to_groq_endpoint(self):
        client = SimpleNamespace(
            chat_completion=Mock(
                return_value=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="  answer  ")
                        )
                    ]
                )
            )
        )

        answer = pdf_summarizer._complete(client, "system", "question")

        self.assertEqual(answer, "answer")
        self.assertEqual(
            client.chat_completion.call_args.kwargs["model"],
            "openai/gpt-oss-20b",
        )
        self.assertEqual(
            client.chat_completion.call_args.kwargs["extra_body"],
            {
                "reasoning_effort": "low",
                "reasoning_format": "hidden",
            },
        )
        self.assertEqual(client.chat_completion.call_args.kwargs["max_tokens"], 1024)

    @patch.dict(os.environ, {}, clear=True)
    def test_completion_allows_a_larger_summary_budget(self):
        client = SimpleNamespace(
            chat_completion=Mock(
                return_value=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="complete summary")
                        )
                    ]
                )
            )
        )

        pdf_summarizer._complete(
            client,
            "system",
            "question",
            max_tokens=2048,
        )

        self.assertEqual(client.chat_completion.call_args.kwargs["max_tokens"], 2048)

    @patch.dict(os.environ, {}, clear=True)
    @patch("pdf_summarizer.time.sleep")
    def test_completion_retries_after_rate_limit(self, sleep):
        response = SimpleNamespace(
            status_code=429,
            headers={},
            request=Mock(),
        )
        rate_limit_error = HfHubHTTPError(
            "Rate limit reached. Please try again in 3.705s.",
            response=response,
        )
        client = SimpleNamespace(
            chat_completion=Mock(
                side_effect=[
                    rate_limit_error,
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="answer")
                            )
                        ]
                    ),
                ]
            )
        )

        answer = pdf_summarizer._complete(client, "system", "question")

        self.assertEqual(answer, "answer")
        self.assertEqual(client.chat_completion.call_count, 2)
        sleep.assert_called_once_with(3.705)

    @patch.dict(os.environ, {}, clear=True)
    def test_create_client_requires_groq_api_key(self):
        with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
            pdf_summarizer.create_client()

    @patch(
        "pdf_summarizer._complete",
        side_effect=["partial summary", "final summary"],
    )
    @patch("pdf_summarizer._extract_pdf_text", return_value="A short PDF excerpt.")
    def test_summarize_reports_progress_through_final_save(
        self, _extract_pdf_text, _complete
    ):
        updates = []
        with tempfile.TemporaryDirectory() as output_folder:
            output_path = pdf_summarizer.summarize_folder(
                Mock(),
                Path("manual.pdf"),
                Path(output_folder),
                progress_callback=lambda fraction, message: updates.append(
                    (fraction, message)
                ),
            )
            self.assertTrue(output_path.is_file())

        self.assertEqual(updates[0][1], "Extracting PDF text...")
        self.assertEqual(updates[-1], (1.0, "Summary saved."))
        self.assertEqual(
            [fraction for fraction, _ in updates],
            sorted(fraction for fraction, _ in updates),
        )


if __name__ == "__main__":
    unittest.main()
