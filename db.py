import sqlite3
import json

DB_PATH = "chatbot.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

# ===============================
# INSERT CHAT
# ===============================
def insert_chat_pair(data: dict):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO chat_pairs (
            conversation_id,
            session_id,
            turn_index,
            user_message,
            admin_response,
            context,
            intent_parent,
            intent_child,
            priority_score,
            reward_count,
            punish_count,
            embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["conversation_id"],
        data["session_id"],
        data["turn_index"],
        data["user_message"],
        data["admin_response"],
        json.dumps(data["context"], ensure_ascii=False),
        data["intent_parent"],
        data["intent_child"],
        data["priority_score"],
        data["reward_count"],
        data["punish_count"],
        json.dumps(data["embedding"])
    ))

    conn.commit()
    conn.close()

# ===============================
# FETCH CONTEXT BY CONVERSATION
# ===============================
def fetch_context(conversation_id, limit=6):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT user_message, admin_response
        FROM chat_pairs
        WHERE conversation_id = ?
        ORDER BY turn_index DESC
        LIMIT ?
    """, (conversation_id, limit))

    rows = cur.fetchall()
    conn.close()

    context = []
    for user_msg, admin_msg in reversed(rows):
        context.append(f"USER:{user_msg}")
        context.append(f"ADMIN:{admin_msg}")

    return context

# ===============================
# FETCH DATASET FOR RETRIEVAL
# ===============================
def fetch_dataset_by_intent(intent_parent=None):
    conn = get_conn()
    cur = conn.cursor()

    if intent_parent:
        cur.execute("""
            SELECT *
            FROM chat_pairs
            WHERE intent_parent = ?
        """, (intent_parent,))
    else:
        cur.execute("SELECT * FROM chat_pairs")

    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    conn.close()

    result = []
    for r in rows:
        item = dict(zip(cols, r))
        item["context"] = json.loads(item["context"]) if item["context"] else []
        item["embedding"] = json.loads(item["embedding"])
        result.append(item)

    return result

def fetch_next_turn_index(conversation_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(MAX(turn_index), 0) + 1
        FROM chat_pairs
        WHERE conversation_id = ?
    """, (conversation_id,))
    val = cur.fetchone()[0]
    conn.close()
    return val

# def insert_payment_event(data):
#     conn = get_conn()
#     c = conn.cursor()

#     c.execute("""
#     INSERT INTO payment_events
#     (conversation_id, session_id, detected_at, source,
#      raw_message, detector_confidence,
#      extracted_name, extracted_amount, extracted_bank)
#     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#     """, (
#         data["conversation_id"],
#         data["session_id"],
#         data["detected_at"],
#         data["source"],
#         data["raw_message"],
#         data["confidence"],
#         data["extracted_name"],
#         data["extracted_amount"],
#         data["extracted_bank"]
#     ))

#     conn.commit()
#     conn.close()

def apply_feedback_db(session_id, rating):
    conn = get_conn()
    cur = conn.cursor()

    if rating == 1:
        cur.execute("""
            UPDATE chat_pairs
            SET priority_score = MIN(priority_score + 5, 100),
                reward_count = reward_count + 1
            WHERE session_id = ?
        """, (session_id,))

    elif rating == -1:
        cur.execute("""
            UPDATE chat_pairs
            SET priority_score = MAX(priority_score - 10, 0),
                punish_count = punish_count + 1
            WHERE session_id = ?
        """, (session_id,))

    conn.commit()

    updated = cur.rowcount  # penting buat validasi
    conn.close()

    return updated > 0

