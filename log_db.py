import json
import sqlite3
from datetime import datetime

LOG_DB_PATH = "log.db"


def _get_conn():
    return sqlite3.connect(LOG_DB_PATH)


def _utc_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _col_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())


def init_log_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            duration_ms INTEGER,
            http_status INTEGER,
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
            placeholder_guard_applied INTEGER,
            admin_response TEXT,
            payload_json TEXT,
            thread_name TEXT
        )
        """
    )

    # Backward compatibility if table was created by old schema.
    required_cols = {
        "request_id": "TEXT UNIQUE",
        "started_at": "TEXT",
        "ended_at": "TEXT",
        "duration_ms": "INTEGER",
        "http_status": "INTEGER",
        "placeholder_guard_applied": "INTEGER",
    }
    for col, col_type in required_cols.items():
        if not _col_exists(cur, "chat_logs", col):
            cur.execute(f"ALTER TABLE chat_logs ADD COLUMN {col} {col_type}")

    # Migrate old created_at data if present.
    if _col_exists(cur, "chat_logs", "created_at"):
        cur.execute(
            """
            UPDATE chat_logs
            SET started_at = COALESCE(started_at, created_at)
            WHERE started_at IS NULL
            """
        )

    conn.commit()
    conn.close()


def start_request_log(request_id, payload=None, thread_name=None, conversation_id=None, user_query=None):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO chat_logs (
            request_id,
            started_at,
            status,
            conversation_id,
            user_query,
            payload_json,
            thread_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            _utc_now(),
            "started",
            conversation_id,
            user_query,
            json.dumps(payload or {}, ensure_ascii=False),
            thread_name,
        ),
    )
    conn.commit()
    conn.close()

def update_request_log(request_id, **fields):
    if not fields:
        return

    col_map = {
        "conversation_id": "conversation_id",
        "session_id": "session_id",
        "status": "status",
        "error_code": "error_code",
        "error_message": "error_message",
        "user_query": "user_query",
        "context_turns": "context_turns",
        "context_text": "context_text",
        "intent_parent": "intent_parent",
        "intent_child": "intent_child",
        "retrieval_candidates": "retrieval_candidates",
        "top_similarity": "top_similarity",
        "top_priority_score": "top_priority_score",
        "top_final_score": "top_final_score",
        "layer1_draft": "layer1_draft",
        "layer2_final": "layer2_final",
        "placeholder_guard_applied": "placeholder_guard_applied",
        "admin_response": "admin_response",
        "thread_name": "thread_name",
        "http_status": "http_status",
        "duration_ms": "duration_ms",
        "ended_at": "ended_at",
        "matches": "matches_json",
        "payload": "payload_json",
    }

    sets = []
    values = []
    for key, value in fields.items():
        col = col_map.get(key)
        if not col:
            continue
        if key in ("matches", "payload"):
            value = json.dumps(value or ([] if key == "matches" else {}), ensure_ascii=False)
        sets.append(f"{col} = ?")
        values.append(value)

    if not sets:
        return

    values.append(request_id)
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        f"UPDATE chat_logs SET {', '.join(sets)} WHERE request_id = ?",
        values,
    )
    conn.commit()
    conn.close()

def finalize_request_log(request_id, status, http_status, **fields):
    fields["status"] = status
    fields["http_status"] = http_status
    fields["ended_at"] = _utc_now()

    col_map = {
        "status": "status",
        "http_status": "http_status",
        "error_code": "error_code",
        "error_message": "error_message",
        "conversation_id": "conversation_id",
        "session_id": "session_id",
        "user_query": "user_query",
        "context_turns": "context_turns",
        "context_text": "context_text",
        "intent_parent": "intent_parent",
        "intent_child": "intent_child",
        "retrieval_candidates": "retrieval_candidates",
        "top_similarity": "top_similarity",
        "top_priority_score": "top_priority_score",
        "top_final_score": "top_final_score",
        "layer1_draft": "layer1_draft",
        "layer2_final": "layer2_final",
        "placeholder_guard_applied": "placeholder_guard_applied",
        "admin_response": "admin_response",
        "payload": "payload_json",
        "matches": "matches_json",
        "thread_name": "thread_name",
        "duration_ms": "duration_ms",
        "ended_at": "ended_at",
    }

    sets = []
    values = []
    for key, value in fields.items():
        col = col_map.get(key)
        if not col:
            continue
        if key in ("matches", "payload"):
            value = json.dumps(value or ([] if key == "matches" else {}), ensure_ascii=False)
        sets.append(f"{col} = ?")
        values.append(value)

    if not sets:
        return

    values.extend([request_id])
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        UPDATE chat_logs
        SET {', '.join(sets)}
        WHERE request_id = ? AND ended_at IS NULL
        """,
        values,
    )
    conn.commit()
    conn.close()
