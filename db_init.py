import sqlite3

DB_PATH = "chatbot.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # =======================
    # CHAT PAIRS
    # =======================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_pairs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        conversation_id INTEGER NOT NULL,
        session_id TEXT,
        turn_index INTEGER NOT NULL,

        user_message TEXT NOT NULL,
        admin_response TEXT NOT NULL,

        context TEXT,

        intent_parent TEXT,
        intent_child TEXT,

        priority_score INTEGER DEFAULT 50,
        reward_count INTEGER DEFAULT 0,
        punish_count INTEGER DEFAULT 0,

        embedding TEXT,

        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_conversation
    ON chat_pairs (conversation_id, turn_index)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_intent_parent
    ON chat_pairs (intent_parent)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_priority
    ON chat_pairs (priority_score DESC)
    """)

    # # =======================
    # # PAYMENT EVENTS (LOG)
    # # =======================
    # cur.execute("""
    # CREATE TABLE IF NOT EXISTS payment_events (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,

    #     conversation_id INTEGER,
    #     session_id TEXT,

    #     detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    #     source TEXT,

    #     raw_message TEXT,

    #     detector_confidence REAL,

    #     extracted_name TEXT,
    #     extracted_amount TEXT,
    #     extracted_bank TEXT,

    #     status TEXT DEFAULT 'pending'
    # )
    # """)

    # # =======================
    # # PAYMENT STATUS (STATE)
    # # =======================
    # cur.execute("""
    # CREATE TABLE IF NOT EXISTS payment_status (
    #     conversation_id INTEGER PRIMARY KEY,
    #     is_paid INTEGER DEFAULT 0,
    #     paid_at TEXT,
    #     confirmed_by TEXT
    # )
    # """)

    conn.commit()
    conn.close()
    print("✅ SQLite DB initialized")

if __name__ == "__main__":
    init_db()
