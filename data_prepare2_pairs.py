import pandas as pd

# Load hasil cleaning
clean_path = "data/cleaned_dataset.json"
df_cleaned = pd.read_json(clean_path)

print("[LOAD] cleaned_dataset.json loaded:", len(df_cleaned), "rows")

# ------------------------
# BUAT PAIRS
# ------------------------
def build_user_admin_pairs(df, context_window=5):
    """
    Build pairs user-admin:
    - context hanya dari conversation_id yang sama
    - context = maksimal 5 chat sebelum user_message
    - context disimpan dalam format berisi role + chat agar embedding memahami percakapan
    """
    print("[PAIRS] Building user-admin pairs with ROLE-AWARE CONTEXT")
    
    # Sort dari awal
    df = df.sort_values(["conversation_id", "created_at"]).reset_index(drop=True)

    pairs = []

    for i in range(len(df) - 1):

        row = df.iloc[i]

        # hanya pair jika baris ini user
        if row["role"] != "user":
            continue

        next_row = df.iloc[i + 1]

        # balasan harus admin & masih dalam conversation ID yang sama
        if (
            next_row["role"] != "admin" or
            next_row["conversation_id"] != row["conversation_id"]
        ):
            continue

        cid = row["conversation_id"]

        # ambil seluruh conversation
        df_conv = df[df["conversation_id"] == cid]

        # posisi row ini dalam conversation (bukan index global)
        pos = df_conv.index.get_loc(i)

        # ambil max 5 pesan sebelum user_message
        df_before = df_conv.iloc[max(0, pos - context_window):pos]

        # === KONTEKS ROLE-AWARE ===
        context_msgs = [
            f"{r['role']} : {r['chat']}"
            for _, r in df_before.iterrows()
        ]

        pairs.append({
            "conversation_id": cid,
            "context": context_msgs, 
            "user_message": row["chat"],
            "admin_response": next_row["chat"],
        })

    df_pairs = pd.DataFrame(pairs)
    print(f"[PAIRS] Total pairs built: {len(df_pairs)}")
    return df_pairs

df_pairs = build_user_admin_pairs(df_cleaned, context_window=5)

# ------------------------
# SAVE PAIRS
# ------------------------
pairs_path = "data/pairs_dataset.json"

df_pairs.to_json(
    pairs_path,
    orient="records",
    indent=2,
    force_ascii=False
)

print("Saved pairs to:", pairs_path)