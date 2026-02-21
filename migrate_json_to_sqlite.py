import json
import sqlite3
from turtle import pd

DB_PATH = "chatbot.db"
JSON_PATH = "model/pairs_perpanjangan_with_intent_embedding.json"

def migrate(overwrite=True):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    if overwrite:
        print("⚠️ Clearing existing chat_pairs data...")
        cur.execute("DELETE FROM chat_pairs")
        conn.commit()

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        import pandas as pd
        df = pd.read_json(JSON_PATH, lines=True)
        data = df.to_dict(orient="records")

    inserted = 0

    for row in data:
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
            row.get("conversation_id", 0),
            row.get("session_id"),
            row.get("turn_index", 0),
            row.get("user_message"),
            row.get("admin_response"),
            json.dumps(row.get("context", []), ensure_ascii=False),
            row.get("intent_parent"),
            row.get("intent_child"),
            row.get("priority_score", 50),
            row.get("reward_count", 0),
            row.get("punish_count", 0),
            json.dumps(row.get("embedding")) if row.get("embedding") else None
        ))

        inserted += 1

    conn.commit()
    conn.close()

    print(f"✅ Migrated {inserted} rows into SQLite")

if __name__ == "__main__":
    migrate(overwrite=True)
