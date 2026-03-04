import json
import sqlite3
from datetime import datetime

LOG_DB_PATH = "log.db"


def _get_conn():
    return sqlite3.connect(LOG_DB_PATH)


def init_log_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            conversation_id INTEGER,
            session_id TEXT,
            status TEXT,
            error_code TEXT,
            error_message TEXT,
            user_query TEXT,
            context_turns INTEGER,
            context_text TEXT,
            intent_parent TEXT,
            intent_child TEXT,
            retrieval_candidates INTEGER,
            top_similarity REAL,
            top_priority_score REAL,
            top_final_score REAL,
            matches_json TEXT,
            layer1_draft TEXT,
            layer2_final TEXT,
            admin_response TEXT,
            payload_json TEXT,
            duration_ms INTEGER,
            thread_name TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_chat_log(data: dict):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_logs (
            created_at,
            conversation_id,
            session_id,
            status,
            error_code,
            error_message,
            user_query,
            context_turns,
            context_text,
            intent_parent,
            intent_child,
            retrieval_candidates,
            top_similarity,
            top_priority_score,
            top_final_score,
            matches_json,
            layer1_draft,
            layer2_final,
            admin_response,
            payload_json,
            duration_ms,
            thread_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            data.get("conversation_id"),
            data.get("session_id"),
            data.get("status"),
            data.get("error_code"),
            data.get("error_message"),
            data.get("user_query"),
            data.get("context_turns"),
            data.get("context_text"),
            data.get("intent_parent"),
            data.get("intent_child"),
            data.get("retrieval_candidates"),
            data.get("top_similarity"),
            data.get("top_priority_score"),
            data.get("top_final_score"),
            json.dumps(data.get("matches") or [], ensure_ascii=False),
            data.get("layer1_draft"),
            data.get("layer2_final"),
            data.get("admin_response"),
            json.dumps(data.get("payload") or {}, ensure_ascii=False),
            data.get("duration_ms"),
            data.get("thread_name"),
        ),
    )
    conn.commit()
    conn.close()
