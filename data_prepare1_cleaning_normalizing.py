import re
import json
from pathlib import Path
from datetime import datetime

RAW_SQL_PATH = "data/clean_dataset.sql"
OUTPUT_JSON_PATH = "data/cleaned_dataset.json"


# ------------------------
# BASIC CLEAN
# ------------------------
def strip_sql_value(v):
    if v is None:
        return None

    s = str(v).strip()

    if s.upper() == "NULL":
        return None

    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1]

    return s.strip()


# ------------------------
# TEXT CLEAN FOR MODEL
# ------------------------
def clean_text_for_model(text):
    if not text:
        return ""

    s = str(text)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-zA-Z0-9\s\.,!?@:/\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()

    return s

# ------------------------
# ROLE NORMALIZATION
# ------------------------
def normalize_roles_basic(role_value):
    if not role_value:
        return "user"

    r = str(role_value).strip().lower()

    if r in ("user", "client"):
        return "user"
    if r in ("admin", "assistant"):
        return "admin"
    if r == "media":
        return "media"
    if r == "system":
        return "system"

    return "user"

# ------------------------
# SQL PARSER
# ------------------------
def parse_sql_values(sql_text):
    inserts = re.findall(
        r"INSERT INTO .*? VALUES\s*(.+?);",
        sql_text,
        flags=re.S | re.I
    )

    rows = []

    for block in inserts:
        tuples = re.findall(r"\((.*?)\)", block, flags=re.S)

        for t in tuples:
            values = []
            buf = ""
            in_string = False

            for ch in t:
                if ch == "'" and not in_string:
                    in_string = True
                    buf += ch
                elif ch == "'" and in_string:
                    in_string = False
                    buf += ch
                elif ch == "," and not in_string:
                    values.append(buf.strip())
                    buf = ""
                else:
                    buf += ch

            if buf:
                values.append(buf.strip())

            rows.append(values)

    return rows

def parse_datetime_safe(v):
    try:
        return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

# ------------------------
# MAIN
# ------------------------
def main():
    sql_text = Path(RAW_SQL_PATH).read_text(
        encoding="utf-8",
        errors="ignore"
    )

    rows = parse_sql_values(sql_text)
    print(f"[LOAD] Raw rows parsed: {len(rows)}")

    cleaned = []

    for r in rows:
        if len(r) < 6:
            continue

        created_at = strip_sql_value(r[1])
        updated_at = strip_sql_value(r[2])

        ts = parse_datetime_safe(created_at) or parse_datetime_safe(updated_at)
        if not ts:
            continue

        conversation_id = strip_sql_value(r[3])
        role_raw = strip_sql_value(r[4])
        chat_raw = strip_sql_value(r[5])

        role = normalize_roles_basic(role_raw)

        # DROP SYSTEM
        if role == "system":
            continue

        chat = clean_text_for_model(chat_raw)
        if not chat:
            continue

        try:
            conversation_id = int(conversation_id)
        except Exception:
            continue

        cleaned.append({
            "created_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "conversation_id": conversation_id,
            "role": role,
            "chat": chat
        })

    # SORT
    cleaned.sort(
        key=lambda x: (x["conversation_id"], x["created_at"])
    )

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {len(cleaned)} rows → {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
