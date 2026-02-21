import pandas as pd
import json

PAIRS_PATH = "data/pairs_dataset.json"
CSV_PATH = "data/perpanjangan_web.csv" 
OUTPUT_JSON_PATH = "data/pairs_perpanjangan.json"

# =========================================================
# LOAD CSV
# =========================================================

def load_perpanjangan_csv(csv_path):
    """
    Membaca file CSV perpanjangan_web.csv dan mengkonversi tipe data kolom numerik.
    """
    print("[CSV] Loading perpanjangan CSV file...")

    try:
        # Gunakan Pandas untuk membaca CSV. low_memory=False disarankan untuk file besar.
        df_csv = pd.read_csv(csv_path, sep=';', encoding='utf-8', low_memory=False)

        # Kolom yang perlu diubah ke tipe numerik
        numeric_cols = [
            "id","user_id","conversation_id","marketing_id","whatsapp_id","price","active_years"
        ]

        # Konversi tipe data
        for col in numeric_cols:
            # 'coerce' akan mengubah nilai yang tidak numerik (termasuk NULL) menjadi NaN
            df_csv[col] = pd.to_numeric(df_csv[col], errors="coerce")

        print("[CSV] Loaded:", len(df_csv), "rows from CSV")
        return df_csv

    except FileNotFoundError:
        print(f"[CSV] ERROR: File tidak ditemukan di jalur: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[CSV] ERROR saat membaca CSV: {e}")
        return pd.DataFrame()
    
# =========================================================
# FILTER df_pairs DENGAN TABEL CSV
# =========================================================

def filter_pairs_by_perpanjangan(df_pairs, df_csv):
    print("[FILTER] Filtering pairs based on perpanjangan_websites...")

    df_pairs["conversation_id"] = pd.to_numeric(
        df_pairs["conversation_id"], errors="coerce"
    )

    perpanjang_ids = set(
        df_csv["conversation_id"].dropna().unique()
    )

    print("[FILTER] Conversation IDs found in CSV:", len(perpanjang_ids))

    df_filtered = df_pairs[
        df_pairs["conversation_id"].isin(perpanjang_ids)
    ].copy()

    print("[FILTER] Pairs before:", len(df_pairs))
    print("[FILTER] Pairs after filtering:", len(df_filtered))

    if df_filtered.empty:
        print("[WARNING] Tidak ada data yang lolos filter!")

    return df_filtered

# =========================================================
# ADD TURN INDEX PER CONVERSATION
# =========================================================

def add_turn_index(df):
    """
    Menambahkan turn_index untuk setiap baris
    berdasarkan urutan kemunculan dalam conversation_id.
    """
    df = df.copy()
    df["turn_index"] = (
        df.groupby("conversation_id")
        .cumcount()
    )
    return df

# =========================================================
# BUILD CONTEXT METADATA (NON-DESTRUCTIVE)
# =========================================================

def add_context_metadata(df):
    """
    Menambahkan:
    - context_text  : gabungan context jadi string
    - context_turns : turn index yang membentuk konteks
    """
    df = df.copy()

    context_texts = []
    context_turns = []

    for _, row in df.iterrows():
        ctx = row.get("context", [])
        if not isinstance(ctx, list):
            ctx = []

        turn_idx = row.get("turn_index", 0)
        start = max(0, turn_idx - len(ctx))

        context_texts.append("\n".join(ctx))
        context_turns.append(list(range(start, turn_idx)))

    df["context_text"] = context_texts
    df["context_turns"] = context_turns

    return df

# =========================================================
# SAVE JSON
# =========================================================

def save_json(obj, path):
    # Menggunakan modul 'json' yang sudah diimport di awal
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("[SAVE] Saved JSON to:", path)

# =========================================================
# MAIN EXECUTION
# =========================================================

def process_perpanjangan_pairs(pairs_path, csv_path, output_json_path):
    # load df_pairs (Tetap dari JSON)
    print(f"Loading pairs dari {pairs_path}...")
    try:
        df_pairs = pd.read_json(pairs_path)
    except Exception as e:
        print(f"ERROR: Gagal memuat df_pairs dari JSON: {e}")
        return pd.DataFrame()

    # load CSV perpanjangan (Menggantikan load_perpanjangan_sql)
    df_csv = load_perpanjangan_csv(csv_path)

    if df_csv.empty:
        print("Proses filtering dihentikan karena data CSV gagal dimuat atau kosong.")
        return pd.DataFrame()

    # filter
    df_filtered = filter_pairs_by_perpanjangan(df_pairs, df_csv)
    df_filtered = add_turn_index(df_filtered)
    df_filtered = add_context_metadata(df_filtered)

    # save json
    save_json(df_filtered.to_dict(orient="records"), output_json_path)

    return df_filtered

# --- Eksekusi Program Utama ---
df_final = process_perpanjangan_pairs(
    pairs_path=PAIRS_PATH,
    csv_path=CSV_PATH,
    output_json_path=OUTPUT_JSON_PATH
)
print("\nProgram selesai dieksekusi.")
if not df_final.empty:
    print(f"Total baris data yang difilter: {len(df_final)}")