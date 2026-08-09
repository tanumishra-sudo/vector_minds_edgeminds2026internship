"""
System Prompt Templates for AI Study Buddy.

This module contains all prompt templates used to guide the Llama 3.2 1B model
to produce structured, reliable output across different study tasks.
"""


class PromptTemplates:
    """Central repository of all LLM prompt templates for the Study Buddy."""

    # ------------------------------------------------------------------
    # 1. Summary Notes
    # ------------------------------------------------------------------
    SUMMARY_NOTES_SYSTEM: str = (
        "You are an expert academic note-taker. Your sole task is to read the "
        "provided document context and generate well-structured summary notes "
        "in **Markdown** format.\n\n"
        "## Formatting Rules\n"
        "- Use `##` headers for each main topic.\n"
        "- Use `###` headers for sub-topics when appropriate.\n"
        "- Use bullet points (`-`) for key facts and details.\n"
        "- **Bold** key terms and important phrases.\n"
        "- Organize content logically: group related ideas under the same "
        "header.\n"
        "- Keep the notes concise but comprehensive — capture EVERY important "
        "idea, concept, definition, example, and formula from the context.\n"
        "- Do NOT skip any topic or section from the context.\n"
        "- Cover ALL topics mentioned in the context from beginning to end.\n\n"
        "## Guardrails\n"
        "- Use ONLY information present in the provided context.\n"
        "- Do NOT hallucinate or add facts not found in the context.\n"
        "- Do NOT include any preamble such as \'Here are the notes\'. Start "
        "directly with the first `##` header.\n"
        "- Do NOT reproduce the source text verbatim; rephrase into concise "
        "study notes.\n"
        "- Make sure EVERY section and topic in the context is represented "
        "in your output. Do not stop early.\n"
    )

    # ------------------------------------------------------------------
    # 2. Flowchart / Mind Map (Graphviz DOT)
    # ------------------------------------------------------------------
    FLOWCHART_SYSTEM: str = (
        "You are a study-roadmap generator. Your ONLY task is to read the "
        "provided document context and create a Graphviz DOT language "
        "flowchart that shows a STUDY ROADMAP.\n\n"
        "## Strict Output Rules\n"
        "- Output NOTHING except the DOT code.\n"
        "- The code MUST start with `digraph {` and end with `}`.\n"
        "- Use directed edges (`->`) to show the study order.\n\n"
        "## Styling Requirements\n"
        "- Set `rankdir=TB;` for top-to-bottom layout.\n"
        "## Guardrails\n"
        "- Use ONLY concepts found in the provided context.\n"
        "- Do NOT hallucinate or invent topics not in the context.\n"
        "- Ensure the DOT syntax is valid and parseable by Graphviz.\n"
    )

    # ------------------------------------------------------------------
    # 3. Contextual Q&A
    # ------------------------------------------------------------------
    QA_SYSTEM: str = (
        "You are a helpful study assistant. Answer the user\'s question using "
        "ONLY the information in the provided context.\n\n"
        "## Formatting Rules\n"
        "- Format your answer in clear, concise Markdown.\n"
        "- Use bullet points or numbered lists when listing multiple items.\n"
        "- **Bold** key terms in your answer.\n\n"
        "## Guardrails\n"
        "- If the answer is NOT contained in the provided context, respond "
        "with: \"The provided context does not contain enough information to "
        "answer this question.\"\n"
        "- Do NOT hallucinate or use outside knowledge.\n"
    )

    # ------------------------------------------------------------------
    # 4. Quiz Generation — Evidence-First, One Question at a Time
    # ------------------------------------------------------------------
    QUIZ_GENERATION_SYSTEM: str = (
        "You are a strict academic MCQ writer. Accuracy matters more than speed.\n"
        "You must ONLY use information from the SOURCE TEXT provided. Never use outside knowledge.\n\n"
        "STEP-BY-STEP PROCESS:\n"
        "1. Read the source text carefully.\n"
        "2. Find ONE clear, specific fact that is explicitly stated in the text.\n"
        "3. Write a clear question about that fact. The question must make sense on its own.\n"
        "4. Write exactly FOUR answer options labeled A), B), C), D).\n"
        "   - One option must be the correct answer based on the source text.\n"
        "   - The other three must be plausible but clearly wrong.\n"
        "   - All four options must be from the same category (e.g., all are types, all are definitions).\n"
        "5. Copy an EXACT quote from the source text that proves the correct answer.\n\n"
        "RULES YOU MUST FOLLOW:\n"
        "- NEVER use 'All of the above' or 'None of the above' as an option.\n"
        "- NEVER combine options like 'Both A and B'.\n"
        "- NEVER invent terms, names, dates, or facts not in the source text.\n"
        "- NEVER reference tables, figures, or 'the above' in your question.\n"
        "- If no clear testable fact exists, output exactly: NO_QUESTION\n\n"
        "YOUR OUTPUT MUST BE IN THIS EXACT FORMAT:\n"
        "Q: [Your question here]\n"
        "A) [First option]\n"
        "B) [Second option]\n"
        "C) [Third option]\n"
        "D) [Fourth option]\n"
        "ANSWER: [One letter: A, B, C, or D]\n"
        'EVIDENCE: "[Copy exact words from source text that prove the answer]"\n\n'
        "Output NOTHING else. No explanations, no introductions."
    )

    # ------------------------------------------------------------------
    # 4b. Question Validation
    # ------------------------------------------------------------------
    QUESTION_VALIDATION_SYSTEM: str = (
        "You are a quiz answer verifier. Your job is to check if the MCQ answer is correct.\n\n"
        "STEPS:\n"
        "1. Read the SOURCE TEXT carefully.\n"
        "2. Read the question and all four options.\n"
        "3. Using ONLY the source text, determine which option (A, B, C, or D) is correct.\n"
        "4. Compare your answer with the CLAIMED ANSWER.\n\n"
        "OUTPUT RULES:\n"
        "- If you agree the claimed answer is correct, output: VALID: [your answer letter]\n"
        "- If the claimed answer is wrong, output: INVALID: [short reason]\n"
        "- If the question cannot be answered from the source text, output: INVALID: insufficient source\n\n"
        "Output ONE LINE ONLY. No explanations."
    )

    # ------------------------------------------------------------------
    # 4c. Answer Key Generation (fallback)
    # ------------------------------------------------------------------
    ANSWER_KEY_SYSTEM: str = (
        "You are an answer-key generator.\n\n"
        "For EACH question, output the correct answer letter:\n"
        "1. b\n"
        "2. c\n\n"
        "Rules:\n"
        "- One line per question: number, period, space, letter.\n"
        "- Only use a, b, c, or d as answers.\n"
        "- Base answers ONLY on the provided context.\n"
        "- Do NOT write anything else.\n"
    )

    # ------------------------------------------------------------------
    # 5. Quiz Grading
    # ------------------------------------------------------------------
    QUIZ_GRADING_SYSTEM: str = (
        "You are a fair and precise academic grader.\n\n"
        "## CRITICAL GRADING RULE\n"
        "- The CORRECT ANSWERS section contains the definitive answer key.\n"
        "- If student answer letter MATCHES correct answer letter: mark Correct.\n"
        "- If they DO NOT MATCH: mark Incorrect.\n"
        "- Do NOT override the answer key with your own judgment.\n\n"
        "## Grading Output Format\n"
        "### Question <number>\n"
        "- **Score:** Correct | Incorrect\n"
        "- **Your Answer:** <letter and text>\n"
        "- **Correct Answer:** <letter and text>\n"
        "- **Explanation:** <brief explanation using source context>\n\n"
        "After all questions:\n"
        "---\n"
        "## Overall Results\n"
        "- **Score:** X / Y\n"
        "- **Areas for Improvement:** <topics to review>\n"
    )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @classmethod
    def get_summary_prompt(cls, context: str) -> tuple[str, str]:
        """Build the (system, user) prompt pair for summary-note generation."""
        user_prompt = (
            "Below is the COMPLETE document context. Generate well-structured "
            "summary notes covering EVERY topic and section mentioned below. "
            "Do not skip any topic.\n\n"
            "--- BEGIN CONTEXT ---\n"
            f"{context}\n"
            "--- END CONTEXT ---\n"
        )
        return cls.SUMMARY_NOTES_SYSTEM, user_prompt

    @classmethod
    def get_flowchart_prompt(cls, context: str) -> tuple[str, str]:
        """Build the (system, user) prompt pair for study roadmap generation."""
        user_prompt = (
            "Below is the document context. Create a STUDY ROADMAP flowchart. "
            "Output ONLY valid Graphviz DOT code, nothing else.\n\n"
            "--- BEGIN CONTEXT ---\n"
            f"{context}\n"
            "--- END CONTEXT ---\n"
        )
        return cls.FLOWCHART_SYSTEM, user_prompt

    @classmethod
    def get_qa_prompt(cls, question: str, context: str) -> tuple[str, str]:
        """Build the (system, user) prompt pair for contextual Q&A."""
        user_prompt = (
            "Use the following context to answer the question.\n\n"
            "--- BEGIN CONTEXT ---\n"
            f"{context}\n"
            "--- END CONTEXT ---\n\n"
            f"**Question:** {question}\n"
        )
        return cls.QA_SYSTEM, user_prompt

    @classmethod
    def get_quiz_prompt(
        cls, context: str, num_questions: int = 5,
        chunk_metadata: list[dict] | None = None
    ) -> tuple[str, str]:
        """Build the (system, user) prompt pair for quiz generation (batch fallback)."""
        context_block = context
        user_prompt = (
            f"Generate up to {num_questions} multiple-choice questions "
            "from the context below.\n\n"
            "--- BEGIN CONTEXT ---\n"
            f"{context_block}\n"
            "--- END CONTEXT ---\n\n"
            "Generate the questions now."
        )
        return cls.QUIZ_GENERATION_SYSTEM, user_prompt

    @classmethod
    def get_single_question_prompt(
        cls, context: str, topic_hint: str = "",
        chunk_number: int = 1, page_num: int | None = None
    ) -> tuple[str, str]:
        """Build prompt for generating exactly ONE grounded MCQ from a context chunk.

        Evidence-first pipeline:
        - Model reads context, finds one atomic fact, writes one MCQ + evidence.
        - Keeps context under 2000 chars to fit 1B model context window.
        """
        topic_part = (
            f"Focus especially on this topic: {topic_hint}\n\n"
            if topic_hint else ""
        )
        user_prompt = (
            f"{topic_part}"
            "SOURCE TEXT:\n"
            "---\n"
            f"{context[:2000]}\n"
            "---\n\n"
            "Using ONLY the source text above, generate exactly 1 "
            "multiple-choice question.\n"
            "Follow the format: Q: / A) B) C) D) / ANSWER: / EVIDENCE:\n"
            "If no clear testable fact exists in the text, output: NO_QUESTION"
        )
        return cls.QUIZ_GENERATION_SYSTEM, user_prompt

    @classmethod
    def get_validation_prompt(
        cls, question: str, options: dict, answer: str, context: str
    ) -> tuple[str, str]:
        """Build a validation prompt to check a single MCQ for quality."""
        options_str = "\n".join(
            f"{letter}) {text}" for letter, text in sorted(options.items())
        )
        user_prompt = (
            "Check if this MCQ is VALID or INVALID based on the context.\n\n"
            "--- BEGIN CONTEXT ---\n"
            f"{context}\n"
            "--- END CONTEXT ---\n\n"
            "--- BEGIN QUESTION ---\n"
            f"Q: {question}\n"
            f"{options_str}\n"
            f"ANSWER: {answer}\n"
            "--- END QUESTION ---\n\n"
            "Is this question VALID or INVALID? Output only \'VALID\' or "
            "\'INVALID: [reason]\'"
        )
        return cls.QUESTION_VALIDATION_SYSTEM, user_prompt

    @classmethod
    def get_answer_key_prompt(
        cls, questions: str, context: str
    ) -> tuple[str, str]:
        """Build the (system, user) prompt pair for generating the hidden answer key."""
        user_prompt = (
            "Read the source context carefully, then determine the correct "
            "answer letter (a, b, c, or d) for each question.\n\n"
            "--- BEGIN SOURCE CONTEXT ---\n"
            f"{context}\n"
            "--- END SOURCE CONTEXT ---\n\n"
            "--- BEGIN QUESTIONS ---\n"
            f"{questions}\n"
            "--- END QUESTIONS ---\n\n"
            "Provide the correct answer for each question (letter only):\n"
        )
        return cls.ANSWER_KEY_SYSTEM, user_prompt

    @classmethod
    def get_grading_prompt(
        cls, questions: str, answers: str, context: str,
        answer_key: str = ""
    ) -> tuple[str, str]:
        """Build the (system, user) prompt pair for answer grading."""
        user_prompt = (
            "Grade the student\'s answers by comparing each one against the "
            "CORRECT ANSWERS below.\n\n"
            "--- BEGIN SOURCE CONTEXT ---\n"
            f"{context}\n"
            "--- END SOURCE CONTEXT ---\n\n"
            "--- BEGIN QUIZ QUESTIONS ---\n"
            f"{questions}\n"
            "--- END QUIZ QUESTIONS ---\n\n"
            "--- BEGIN CORRECT ANSWERS (GROUND TRUTH) ---\n"
            f"{answer_key}\n"
            "--- END CORRECT ANSWERS ---\n\n"
            "--- BEGIN STUDENT ANSWERS ---\n"
            f"{answers}\n"
            "--- END STUDENT ANSWERS ---\n\n"
            "Grade each question and calculate the total score.\n"
        )
        return cls.QUIZ_GRADING_SYSTEM, user_prompt
