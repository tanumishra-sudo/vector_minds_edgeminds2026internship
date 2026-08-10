"""
LLM Engine — Phase 3: Structural Interface for AI Study Buddy
==============================================================

Interconnects with a local Ollama server running Llama 3.2 1B to power
all generative features of the study-buddy pipeline:

* Summary-note generation (clean Markdown)
* Flowchart generation (Graphviz DOT syntax)
* RAG-augmented Q&A
* Quiz generation & grading

The module exposes both **streaming** and **non-streaming** response modes
and includes validation helpers that extract and sanitise structured output
(Markdown / DOT) from raw LLM responses.

Dependencies
------------
- ``ollama``  — Python client for the Ollama REST API.

Usage
-----
>>> from core.llm_engine import LLMEngine
>>> engine = LLMEngine()
>>> if engine._check_connection():
...     answer = engine.generate("Explain photosynthesis in 3 bullet points.")
...     print(answer)
"""

from __future__ import annotations

import re
import logging
from typing import Generator

try:
    import ollama
except ImportError:
    ollama = None  # Handled gracefully at runtime

logger = logging.getLogger(__name__)


class LLMEngine:
    """Central interface to the local Ollama LLM backend.

    All public ``generate_*`` methods accept a *system_prompt* that is
    expected to come from the prompt-template layer (Phase 4).  The engine
    itself is prompt-agnostic — it simply forwards the messages to the model
    and post-processes the output.

    Parameters
    ----------
    model_name : str, optional
        Ollama model tag to use (default: ``"llama3.2:1b"``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, model_name: str = "llama3.2:1b") -> None:
        """Initialise the LLM engine.

        Parameters
        ----------
        model_name : str
            The Ollama model identifier, e.g. ``"llama3.2:1b"``.
        """
        self.model_name: str = model_name
        logger.info("LLMEngine initialised with model '%s'", self.model_name)

    # ------------------------------------------------------------------
    # Connection health-check
    # ------------------------------------------------------------------

    def _check_connection(self) -> bool:
        """Check whether the Ollama server is reachable and the configured
        model is available locally.

        Returns
        -------
        bool
            ``True`` if the server responds **and** the model tag is found
            in the local model list; ``False`` otherwise.
        """
        if ollama is None:
            logger.error(
                "The 'ollama' Python package is not installed. "
                "Install it with: pip install ollama"
            )
            return False

        try:
            model_list = ollama.list()

            # ── Extract model names (handles ALL ollama package versions) ──
            available_models: list[str] = []

            # Newer ollama (>=0.4): ListResponse with .models list of Model objects
            # Each Model has .model attribute (the tag string)
            # Older ollama: dict with "models" key, each entry is a dict with "name"
            if isinstance(model_list, dict):
                models_data = model_list.get("models", [])
            else:
                models_data = getattr(model_list, "models", [])

            for model_info in models_data:
                # Try all known attribute names across versions
                if isinstance(model_info, dict):
                    name = model_info.get("name", "") or model_info.get("model", "")
                else:
                    name = (
                        getattr(model_info, "model", "")
                        or getattr(model_info, "name", "")
                    )
                if name:
                    available_models.append(name)

            logger.info("Available Ollama models: %s", available_models)

            # Match flexibly — allow "llama3.2:1b" to match "llama3.2:1b"
            # as well as potential tag variations.
            target = self.model_name.lower()
            found = any(
                target in m.lower() or m.lower().startswith(target.split(":")[0])
                for m in available_models
            )

            if not found:
                logger.warning(
                    "Model '%s' not found among local models: %s. "
                    "Pull it with: ollama pull %s",
                    self.model_name,
                    available_models,
                    self.model_name,
                )
                return False

            logger.info("Ollama connection OK — model '%s' is available.", self.model_name)
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error("Cannot reach Ollama server: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Send a prompt to the Ollama model and return the response.

        Parameters
        ----------
        prompt : str
            The user-facing prompt / question.
        system_prompt : str, optional
            An optional system-level instruction that shapes LLM behaviour.
        stream : bool, optional
            If ``True``, return a **generator** that yields response chunks
            as they arrive (useful for live-display in Streamlit).
            If ``False`` (default), block until the full response is ready
            and return it as a single string.

        Returns
        -------
        str | Generator[str, None, None]
            Complete response text (``stream=False``) or a chunk generator
            (``stream=True``).

        Raises
        ------
        RuntimeError
            If the ``ollama`` package is missing.
        """
        if ollama is None:
            return self._missing_ollama_msg()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if stream:
            return self._stream_response(messages)
        return self._blocking_response(messages)

    # ------------------------------------------------------------------
    # Internal streaming / blocking helpers
    # ------------------------------------------------------------------

    def _blocking_response(self, messages: list[dict[str, str]]) -> str:
        """Call Ollama in blocking mode and return the full response text."""
        options = {"num_ctx": 2048, "num_thread": 4, "temperature": 0.3}
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                options=options,
            )
            content = (
                response.get("message", {}).get("content", "")
                if isinstance(response, dict)
                else getattr(getattr(response, "message", None), "content", "")
            )
            return content.strip()

        except ollama.ResponseError as exc:
            logger.warning("Ollama response error with num_ctx=2048: %s. Retrying with num_ctx=1024...", exc)
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    options={"num_ctx": 1024, "num_thread": 2, "temperature": 0.3},
                )
                content = (
                    response.get("message", {}).get("content", "")
                    if isinstance(response, dict)
                    else getattr(getattr(response, "message", None), "content", "")
                )
                return content.strip()
            except Exception as retry_exc:
                logger.error("Ollama retry failed: %s", retry_exc)
                return f"⚠️ **LLM Error:** The model returned an error — {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama connection failure: %s", exc)
            return (
                "⚠️ **Connection Error:** Could not reach the Ollama server. "
                "Make sure it is running (`ollama serve`) and try again.\n\n"
                f"_Details: {exc}_"
            )

    def _stream_response(
        self, messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Call Ollama in streaming mode and yield text chunks."""
        options = {"num_ctx": 2048, "num_thread": 4, "temperature": 0.3}
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options=options,
            )
            for chunk in stream:
                token = (
                    chunk.get("message", {}).get("content", "")
                    if isinstance(chunk, dict)
                    else getattr(getattr(chunk, "message", None), "content", "")
                )
                if token:
                    yield token

        except ollama.ResponseError as exc:
            logger.warning("Ollama streaming error: %s. Falling back to non-streaming response...", exc)
            try:
                fallback_text = self._blocking_response(messages)
                yield fallback_text
            except Exception as fallback_exc:
                yield f"\n\n⚠️ **LLM Error:** {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama streaming connection failure: %s", exc)
            yield (
                "\n\n⚠️ **Connection Error:** Lost connection to Ollama. "
                f"_Details: {exc}_"
            )

    # ------------------------------------------------------------------
    # High-level generation methods
    # ------------------------------------------------------------------

    def generate_summary_notes(self, context: str, system_prompt: str) -> str:
        """Generate structured summary notes from the supplied context.

        The *system_prompt* (from the prompt-template layer) instructs the
        LLM to output clean, hierarchical **Markdown** notes.

        Parameters
        ----------
        context : str
            Source material to summarise (extracted document text).
        system_prompt : str
            System instruction enforcing Markdown note format.

        Returns
        -------
        str
            Markdown-formatted summary notes.
        """
        user_prompt = (
            "Using the following source material, produce structured study notes.\n\n"
            "--- SOURCE MATERIAL ---\n"
            f"{context}\n"
            "--- END ---"
        )
        response = self.generate(prompt=user_prompt, system_prompt=system_prompt)
        return self._clean_markdown(response)

    def generate_flowchart(self, context: str, system_prompt: str) -> str:
        """Generate a Graphviz DOT-language flowchart from context.

        The *system_prompt* must instruct the LLM to output **only** valid
        DOT code.  This method additionally post-processes the response to
        extract and validate the DOT block.

        Parameters
        ----------
        context : str
            Source material describing a process or concept.
        system_prompt : str
            System instruction enforcing DOT output format.

        Returns
        -------
        str
            A validated Graphviz DOT string, or an error message beginning
            with ``"ERROR:"`` if extraction / validation fails.
        """
        user_prompt = (
            "Create a flowchart in Graphviz DOT language for the following content.\n\n"
            "--- CONTENT ---\n"
            f"{context}\n"
            "--- END ---"
        )
        response = self.generate(prompt=user_prompt, system_prompt=system_prompt)
        return self._extract_dot_code(response)

    def generate_qa_answer(
        self, question: str, context: str, system_prompt: str
    ) -> str:
        """Answer a question using the provided RAG context.

        Parameters
        ----------
        question : str
            The student's question.
        context : str
            Relevant document chunks retrieved by the RAG pipeline.
        system_prompt : str
            System instruction for answer formatting.

        Returns
        -------
        str
            Markdown-formatted answer.
        """
        user_prompt = (
            f"**Question:** {question}\n\n"
            "Use the following context to answer the question. "
            "If the context does not contain the answer, say so clearly.\n\n"
            "--- CONTEXT ---\n"
            f"{context}\n"
            "--- END ---"
        )
        response = self.generate(prompt=user_prompt, system_prompt=system_prompt)
        return self._clean_markdown(response)

    def generate_quiz(
        self, context: str, system_prompt: str, num_questions: int = 5
    ) -> str:
        """Generate quiz questions from the supplied context.

        Parameters
        ----------
        context : str
            Source material to base quiz questions on.
        system_prompt : str
            System instruction specifying quiz format.
        num_questions : int, optional
            Desired number of quiz questions (default 5).

        Returns
        -------
        str
            Formatted quiz (Markdown).
        """
        user_prompt = (
            f"Generate exactly {num_questions} quiz questions based on the "
            "following material.\n\n"
            "--- MATERIAL ---\n"
            f"{context}\n"
            "--- END ---"
        )
        response = self.generate(prompt=user_prompt, system_prompt=system_prompt)
        return self._clean_markdown(response)

    def validate_quiz_question(self, source_text: str, candidate: str, system_prompt: str) -> str:
        """Independently verify one MCQ against its exact source chunk."""
        if ollama is None:
            return "INVALID: Ollama unavailable"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content":
            "--- SOURCE TEXT ---\n" + source_text[:3000] +
            "\n--- END SOURCE ---\n\n--- CANDIDATE MCQ ---\n" + candidate +
            "\n--- END CANDIDATE ---"})
        try:
            response = ollama.chat(
                model=self.model_name, messages=messages, stream=False,
                options={"temperature": 0, "top_p": 0.1, "num_ctx": 2048, "num_thread": 4}
            )
            content = (response.get("message", {}).get("content", "")
                       if isinstance(response, dict)
                       else getattr(getattr(response, "message", None), "content", ""))
            return content.strip()
        except Exception as exc:
            logger.warning("Quiz validation failed: %s", exc)
            return f"INVALID: validator error: {exc}"


    def grade_quiz(
        self,
        questions: str,
        answers: str,
        context: str,
        system_prompt: str,
    ) -> str:
        """Grade student quiz answers against the original context.

        Parameters
        ----------
        questions : str
            The quiz questions that were presented.
        answers : str
            The student's answers.
        context : str
            Original source material for fact-checking.
        system_prompt : str
            System instruction specifying grading rubric.

        Returns
        -------
        str
            Markdown-formatted feedback with per-question scores and
            an overall assessment.
        """
        user_prompt = (
            "Grade the following quiz answers. For each question, indicate "
            "whether the answer is correct, partially correct, or incorrect, "
            "and provide a brief explanation.\n\n"
            "--- QUESTIONS ---\n"
            f"{questions}\n\n"
            "--- STUDENT ANSWERS ---\n"
            f"{answers}\n\n"
            "--- REFERENCE MATERIAL ---\n"
            f"{context}\n"
            "--- END ---"
        )
        response = self.generate(prompt=user_prompt, system_prompt=system_prompt)
        return self._clean_markdown(response)

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def _extract_dot_code(self, response: str) -> str:
        """Extract and validate a Graphviz DOT block from the LLM response.

        The method tries, in order:

        1. Fenced code block marked ``dot`` or ``graphviz``.
        2. Fenced code block without a language tag that *looks like* DOT.
        3. The raw response itself, if it begins with ``digraph`` or ``graph``.

        Basic structural validation ensures the extracted string contains
        ``digraph`` or ``graph`` and balanced braces.

        Parameters
        ----------
        response : str
            Raw LLM output potentially containing a DOT code block.

        Returns
        -------
        str
            Extracted DOT code, or an error string prefixed with
            ``"ERROR:"`` if nothing valid can be found.
        """
        if not response or response.startswith("⚠️"):
            return response or "ERROR: Empty response from LLM."

        # 1. Try fenced code blocks with explicit language tag.
        pattern_tagged = re.compile(
            r"```(?:dot|graphviz)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
        )
        match = pattern_tagged.search(response)
        if match:
            dot_code = match.group(1).strip()
            if self._validate_dot(dot_code):
                return dot_code

        # 2. Try any fenced code block whose content looks like DOT.
        pattern_generic = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
        for match in pattern_generic.finditer(response):
            candidate = match.group(1).strip()
            if self._validate_dot(candidate):
                return candidate

        # 3. Fallback — treat entire response as potential DOT.
        stripped = response.strip()
        if self._validate_dot(stripped):
            return stripped

        logger.warning("Could not extract valid DOT code from LLM response.")
        return (
            "ERROR: The LLM response did not contain valid Graphviz DOT code. "
            "Try regenerating the flowchart."
        )

    @staticmethod
    def _validate_dot(code: str) -> bool:
        """Run lightweight structural checks on a DOT string.

        Checks performed:

        * Contains ``digraph`` or ``graph`` keyword.
        * Opening and closing braces are balanced.

        Parameters
        ----------
        code : str
            Candidate DOT code.

        Returns
        -------
        bool
            ``True`` if the code passes all checks.
        """
        if not code:
            return False

        lower = code.lower()
        has_keyword = "digraph" in lower or "graph" in lower
        braces_balanced = code.count("{") > 0 and code.count("{") == code.count("}")
        return has_keyword and braces_balanced

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Lightly sanitise Markdown output from the LLM.

        * Strips leading/trailing whitespace.
        * Removes residual system-prompt leakage (common with small models).
        * Normalises excessive blank lines to at most two consecutive.

        Parameters
        ----------
        text : str
            Raw LLM Markdown output.

        Returns
        -------
        str
            Cleaned Markdown string.
        """
        if not text:
            return ""
        # Remove any accidental repetition of the system instruction.
        text = text.strip()
        # Collapse runs of 3+ blank lines into 2.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _missing_ollama_msg() -> str:
        """Return a user-friendly error when the ``ollama`` package is absent."""
        return (
            "⚠️ **Missing Dependency:** The `ollama` Python package is not "
            "installed.\n\n"
            "Install it with:\n"
            "```\npip install ollama\n```"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"LLMEngine(model_name={self.model_name!r})" 
