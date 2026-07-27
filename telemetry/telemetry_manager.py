# ============================================================
# telemetry/telemetry_manager.py
# Phase 5: Local Persistence - SQLite Telemetry Engine
# ============================================================
# Preserves progress tracking parameters within an embedded
# SQLite engine. Tracks document uploads, feature usage,
# quiz scores, and session metrics.
# ============================================================

import sqlite3
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Default database path
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(DB_DIR, "study_telemetry.db")


class TelemetryManager:
    """
    Local SQLite-based telemetry engine for tracking student
    interactions, document processing metrics, and quiz performance.
    
    Phase 5 component: Zero-configuration native file-driven
    embedded engine for system telemetry and logging metrics.
    """

    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize telemetry manager with SQLite database.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_database()
        logger.info(f"TelemetryManager initialized with database at: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_database(self):
        """Create all required tables if they do not exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # ── Document Upload Tracking ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_size_bytes INTEGER,
                num_pages INTEGER,
                num_chunks INTEGER,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processing_time_ms REAL
            )
        """)

        # ── Feature Usage Tracking ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                feature_name TEXT NOT NULL,
                usage_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                response_time_ms REAL,
                success INTEGER DEFAULT 1,
                error_message TEXT,
                FOREIGN KEY (document_id) REFERENCES document_uploads(id)
            )
        """)

        # ── Quiz Performance Tracking ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                num_questions INTEGER,
                score_percent REAL,
                quiz_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                weak_areas TEXT,
                FOREIGN KEY (document_id) REFERENCES document_uploads(id)
            )
        """)

        # ── Q&A Session Tracking ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                question TEXT NOT NULL,
                answer_length INTEGER,
                session_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES document_uploads(id)
            )
        """)

        # ── System Performance Metrics ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_unit TEXT,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Database tables initialized successfully.")

    # ── Document Tracking ──────────────────────────────────

    def log_document_upload(
        self,
        filename: str,
        file_size_bytes: int,
        num_pages: int,
        num_chunks: int,
        processing_time_ms: float
    ) -> int:
        """
        Log a document upload event.
        
        Returns:
            The document_id of the inserted record.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO document_uploads 
               (filename, file_size_bytes, num_pages, num_chunks, processing_time_ms)
               VALUES (?, ?, ?, ?, ?)""",
            (filename, file_size_bytes, num_pages, num_chunks, processing_time_ms)
        )
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Logged document upload: {filename} (ID: {doc_id})")
        return doc_id

    # ── Feature Usage Tracking ─────────────────────────────

    def log_feature_usage(
        self,
        document_id: int,
        feature_name: str,
        response_time_ms: float,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log usage of a feature (summary, flowchart, Q&A, quiz)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO feature_usage 
               (document_id, feature_name, response_time_ms, success, error_message)
               VALUES (?, ?, ?, ?, ?)""",
            (document_id, feature_name, response_time_ms, int(success), error_message)
        )
        conn.commit()
        conn.close()
        logger.info(f"Logged feature usage: {feature_name} for doc {document_id}")

    # ── Quiz Results Tracking ──────────────────────────────

    def log_quiz_result(
        self,
        document_id: int,
        num_questions: int,
        score_percent: float,
        weak_areas: str = ""
    ):
        """Log a quiz result with score and identified weak areas."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO quiz_results 
               (document_id, num_questions, score_percent, weak_areas)
               VALUES (?, ?, ?, ?)""",
            (document_id, num_questions, score_percent, weak_areas)
        )
        conn.commit()
        conn.close()
        logger.info(f"Logged quiz result: {score_percent}% for doc {document_id}")

    # ── Q&A Session Tracking ───────────────────────────────

    def log_qa_session(
        self,
        document_id: int,
        question: str,
        answer_length: int
    ):
        """Log a Q&A interaction."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO qa_sessions 
               (document_id, question, answer_length)
               VALUES (?, ?, ?)""",
            (document_id, question, answer_length)
        )
        conn.commit()
        conn.close()

    # ── System Metrics ─────────────────────────────────────

    def log_system_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_unit: str = ""
    ):
        """Log a system performance metric."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO system_metrics 
               (metric_name, metric_value, metric_unit)
               VALUES (?, ?, ?)""",
            (metric_name, metric_value, metric_unit)
        )
        conn.commit()
        conn.close()

    # ── Analytics Queries ──────────────────────────────────

    def get_total_documents(self) -> int:
        """Get total number of documents uploaded."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM document_uploads")
        result = cursor.fetchone()["count"]
        conn.close()
        return result

    def get_recent_documents(self, limit: int = 10) -> list[dict]:
        """Get the most recently uploaded documents."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM document_uploads ORDER BY upload_timestamp DESC LIMIT ?",
            (limit,)
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_feature_usage_stats(self) -> dict:
        """Get aggregated feature usage statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT feature_name, 
                   COUNT(*) as total_uses,
                   AVG(response_time_ms) as avg_response_ms,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures
            FROM feature_usage 
            GROUP BY feature_name
        """)
        results = {row["feature_name"]: dict(row) for row in cursor.fetchall()}
        conn.close()
        return results

    def get_quiz_performance_history(self, document_id: Optional[int] = None) -> list[dict]:
        """Get quiz performance history, optionally filtered by document."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if document_id:
            cursor.execute(
                "SELECT * FROM quiz_results WHERE document_id = ? ORDER BY quiz_timestamp DESC",
                (document_id,)
            )
        else:
            cursor.execute("SELECT * FROM quiz_results ORDER BY quiz_timestamp DESC")
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_average_quiz_score(self) -> float:
        """Get average quiz score across all attempts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(score_percent) as avg_score FROM quiz_results")
        result = cursor.fetchone()["avg_score"]
        conn.close()
        return result if result is not None else 0.0

    def get_study_session_summary(self) -> dict:
        """Get a comprehensive summary of all study activity."""
        return {
            "total_documents": self.get_total_documents(),
            "feature_usage": self.get_feature_usage_stats(),
            "average_quiz_score": self.get_average_quiz_score(),
            "recent_documents": self.get_recent_documents(5),
            "quiz_history": self.get_quiz_performance_history(),
        }
