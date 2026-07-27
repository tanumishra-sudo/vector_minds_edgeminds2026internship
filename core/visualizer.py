"""
Flowchart Visualizer — Graphviz Core Integration
==================================================

Compiles DOT language syntax into PNG images using the ``graphviz`` Python
library.  Designed for use inside a Streamlit front-end: every public method
either returns raw PNG **bytes** ready for ``st.image()`` or a validated DOT
string that can be fed back into the rendering pipeline.

Key responsibilities
--------------------
* Render DOT → PNG and return the image as ``bytes``.
* Validate DOT syntax with a dry-run render (no file written).
* Sanitize common DOT issues produced by LLM outputs.
* Supply a safe default flowchart when the LLM fails entirely.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Optional, Tuple

import graphviz  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class FlowchartVisualizer:
    """Compile DOT language strings into PNG images via Graphviz.

    Parameters
    ----------
    output_dir : str, optional
        Directory used for temporary render artefacts.  Created automatically
        if it does not already exist.  Defaults to ``"cache"``.

    Examples
    --------
    >>> viz = FlowchartVisualizer()
    >>> dot = viz.create_default_flowchart("My Doc")
    >>> png_bytes = viz.render_dot_to_png(dot)
    >>> assert png_bytes is not None
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, output_dir: str = "cache") -> None:
        """Initialise the visualizer with an output directory for temp files.

        Parameters
        ----------
        output_dir : str
            Path (absolute or relative) where intermediate Graphviz files are
            stored during rendering.  The directory is created if absent.
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("FlowchartVisualizer initialised — output_dir=%s", self.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_dot_to_png(
        self,
        dot_source: str,
        filename: str = "flowchart",
    ) -> Optional[bytes]:
        """Render a DOT language string to a PNG image and return raw bytes.

        The method writes a temporary file inside ``self.output_dir``, asks
        Graphviz to compile it to PNG, reads the result back as ``bytes``,
        and returns them.  The caller can pass these bytes straight to
        ``st.image()`` in Streamlit.

        Parameters
        ----------
        dot_source : str
            A valid DOT language string (e.g. ``"digraph { A -> B }"``).
        filename : str, optional
            Base name (without extension) used for the intermediate file.
            Defaults to ``"flowchart"``.

        Returns
        -------
        bytes or None
            Raw PNG image data on success, or ``None`` when rendering fails
            (the error is logged rather than raised).
        """
        try:
            source = graphviz.Source(
                dot_source,
                directory=self.output_dir,
                filename=filename,
                format="png",
            )
            # render() returns the path to the output file (without the
            # format extension appended when `outfile` is not given).
            rendered_path: str = source.render(cleanup=True)

            # graphviz.Source.render returns the path *with* extension.
            if not os.path.isfile(rendered_path):
                # Older graphviz versions may omit the extension — try both.
                rendered_path_with_ext = f"{rendered_path}.png"
                if os.path.isfile(rendered_path_with_ext):
                    rendered_path = rendered_path_with_ext
                else:
                    logger.error(
                        "Rendered file not found at %s (or %s)",
                        rendered_path,
                        rendered_path_with_ext,
                    )
                    return None

            with open(rendered_path, "rb") as fh:
                png_bytes = fh.read()

            logger.info(
                "Rendered DOT to PNG (%d bytes) — %s",
                len(png_bytes),
                rendered_path,
            )
            return png_bytes

        except graphviz.backend.execute.ExecutableNotFound:
            logger.error(
                "Graphviz executable not found.  Please install Graphviz "
                "(https://graphviz.org/download/) and ensure `dot` is on PATH."
            )
            return None
        except graphviz.CalledProcessError as exc:
            logger.error("Graphviz rendering failed: %s", exc)
            return None
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error while rendering DOT to PNG")
            return None

    def validate_dot_syntax(self, dot_source: str) -> Tuple[bool, str]:
        """Check whether *dot_source* is syntactically valid DOT.

        Validation is performed by attempting a dry-run render inside a
        temporary directory.  No artefact files are kept on disk.

        Parameters
        ----------
        dot_source : str
            The DOT language string to validate.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` when the syntax is valid, or
            ``(False, "<error description>")`` when it is not.
        """
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                source = graphviz.Source(
                    dot_source,
                    directory=tmp_dir,
                    filename="_validation",
                    format="png",
                )
                source.render(cleanup=True)
            return True, ""
        except graphviz.backend.execute.ExecutableNotFound:
            msg = (
                "Graphviz executable not found — cannot validate.  "
                "Install Graphviz and ensure `dot` is on PATH."
            )
            logger.error(msg)
            return False, msg
        except graphviz.CalledProcessError as exc:
            error_msg = str(exc).strip()
            logger.warning("DOT validation failed: %s", error_msg)
            return False, error_msg
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Unexpected validation error: {exc}"
            logger.exception(error_msg)
            return False, error_msg

    def create_default_flowchart(
        self,
        title: str = "Document Structure",
        topics: list[str] | None = None,
    ) -> str:
        """Return a fallback DOT string built from actual document topics.

        If *topics* is provided, builds a study roadmap from those topics.
        Otherwise falls back to a generic pipeline chart.

        Parameters
        ----------
        title : str, optional
            The graph label shown at the top of the flowchart.
        topics : list[str] or None
            List of topic strings extracted from the document.

        Returns
        -------
        str
            A syntactically valid DOT ``digraph`` string.
        """
        escaped_title = title.replace('"', '\\"')

        if not topics or len(topics) < 2:
            # Generic fallback only if we truly have nothing
            return (
                f'digraph default_flow {{\n'
                f'    labelloc="t";\n'
                f'    label="{escaped_title}";\n'
                f'    fontname="Helvetica";\n'
                f'    fontsize=16;\n'
                f'    rankdir=TB;\n'
                f'\n'
                f'    node [\n'
                f'        shape=box,\n'
                f'        style="rounded,filled",\n'
                f'        fillcolor="#E8F0FE",\n'
                f'        fontname="Helvetica",\n'
                f'        fontsize=12\n'
                f'    ];\n'
                f'\n'
                f'    edge [\n'
                f'        color="#4285F4",\n'
                f'        penwidth=1.5\n'
                f'    ];\n'
                f'\n'
                f'    upload  [label="Upload\\nDocument"];\n'
                f'    parse   [label="Parse\\nContent"];\n'
                f'    summary [label="Summarize\\nKey Points"];\n'
                f'    quiz    [label="Generate\\nQuiz"];\n'
                f'\n'
                f'    upload -> parse -> summary -> quiz;\n'
                f'}}\n'
            )

        # Build a topic-based study roadmap
        # Color scheme: green for first third, yellow for middle, red for last
        colors = []
        n = len(topics)
        for i in range(n):
            if i < n / 3:
                colors.append('#d4edda')  # Basic (green)
            elif i < 2 * n / 3:
                colors.append('#fff3cd')  # Intermediate (yellow)
            else:
                colors.append('#f8d7da')  # Advanced (red)

        lines = [
            f'digraph study_roadmap {{',
            f'    labelloc="t";',
            f'    label="Study Roadmap: {escaped_title}";',
            f'    fontname="Helvetica";',
            f'    fontsize=16;',
            f'    rankdir=TB;',
            f'',
            f'    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
            f'    edge [color="#4285F4", penwidth=1.5];',
            f'',
            f'    start [label="START", shape=ellipse, fillcolor="#667eea", fontcolor="white"];',
        ]

        # Add topic nodes
        for i, topic in enumerate(topics):
            safe_label = topic.replace('"', '\\"').replace('\n', '\\n')
            node_id = f'topic_{i}'
            lines.append(
                f'    {node_id} [label="{safe_label}", fillcolor="{colors[i]}"];'
            )

        # Add edges: start -> topic_0 -> topic_1 -> ... -> end
        lines.append('')
        lines.append(f'    start -> topic_0 [label="begin with"];')
        for i in range(len(topics) - 1):
            lines.append(f'    topic_{i} -> topic_{i+1} [label="then study"];')

        lines.append(f'')
        lines.append(f'    finish [label="COMPLETE", shape=ellipse, fillcolor="#667eea", fontcolor="white"];')
        lines.append(f'    topic_{len(topics)-1} -> finish;')
        lines.append(f'}}')

        return '\n'.join(lines) + '\n'

    def sanitize_dot_source(self, dot_source: str) -> str:
        """Clean up common DOT syntax problems in LLM-generated output.

        The sanitiser applies the following corrections **in order**:

        1. Strip Markdown fenced-code-block wrappers (` ```dot … ``` `).
        2. Ensure the source is wrapped in a ``digraph { … }`` block.
        3. Append missing semicolons at the end of statement lines.
        4. Escape un-escaped double-quotes inside node/edge labels.
        5. Normalise whitespace (collapse blank lines, strip trailing spaces).

        Parameters
        ----------
        dot_source : str
            Raw DOT string, possibly with LLM artefacts.

        Returns
        -------
        str
            A cleaned-up DOT string that is more likely to compile
            successfully.
        """
        source = dot_source.strip()

        # 1. Strip Markdown fenced code block markers.
        source = re.sub(
            r"^```(?:dot|graphviz|gv)?\s*\n?", "", source, flags=re.MULTILINE
        )
        source = re.sub(r"\n?```\s*$", "", source, flags=re.MULTILINE)
        source = source.strip()

        # 2. Ensure a digraph wrapper exists.
        if not re.match(r"^\s*(strict\s+)?(di)?graph\s", source, re.IGNORECASE):
            source = f"digraph G {{\n{source}\n}}"
            logger.debug("Wrapped bare DOT statements in a digraph block.")

        # 3. Append missing semicolons.
        #    For each non-blank line that is not a brace-only line and does
        #    not already end with ';', '{', or '}', add a semicolon.
        lines = source.split("\n")
        fixed_lines: list[str] = []
        for line in lines:
            stripped = line.rstrip()
            if stripped and not stripped.endswith((";", "{", "}", ",")):
                # Do not touch lines that are purely comments.
                if not stripped.lstrip().startswith("//"):
                    stripped += ";"
            fixed_lines.append(stripped)
        source = "\n".join(fixed_lines)

        # 4. Escape unescaped quotes inside label="…" values.
        #    Strategy: find label="<content>" and escape interior quotes that
        #    are not already preceded by a backslash.
        def _escape_label_quotes(match: re.Match[str]) -> str:
            prefix = match.group(1)  # e.g. 'label="'
            content = match.group(2)
            suffix = match.group(3)  # closing '"'
            # Escape any quote inside content that is not already escaped.
            content = re.sub(r'(?<!\\)"', '\\"', content)
            return f"{prefix}{content}{suffix}"

        source = re.sub(
            r'(label\s*=\s*")(.*?)(")',
            _escape_label_quotes,
            source,
            flags=re.DOTALL,
        )

        # 5. Normalise whitespace.
        source = re.sub(r"\n{3,}", "\n\n", source)  # collapse excess blanks
        source = re.sub(r"[ \t]+$", "", source, flags=re.MULTILINE)  # trailing ws
        source = source.strip() + "\n"

        logger.debug("Sanitised DOT source (%d chars).", len(source))
        return source
