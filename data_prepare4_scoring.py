import pandas as pd
import json
import re
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# LOAD FILTERED PAIRS (HASIL CLEANING + GROUPING)
# =========================================================

PAIR_FILTERED_PATH = "data/pairs_perpanjangan.json"
OUTPUT_PATH = "data/pairs_perpanjangan_with_intent_and_score.json"
MAX_WORKERS = 6 

print("[LOAD] Loading filtered pairs...")
df_filtered = pd.read_json(PAIR_FILTERED_PATH)
print("[LOAD] Loaded:", len(df_filtered), "rows")

# =========================================================
# SETUP OPENAI CLIENT
# =========================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_int(raw):
    m = re.search(r"\b([0-9]{1,3})\b", raw)
    return int(m.group(1)) if m else 0

# =========================================================
# CONTEXT BUILDER FOR GPT
# =========================================================

def build_gpt_context(row, max_turns=5):
    ctx = row.get("context", [])
    if not isinstance(ctx, list):
        return ""
    
    ctx = ctx[-max_turns:]
    return "\n".join([f"- {c}" for c in ctx])
# =========================================================
# INTENT CLASSIFIER
# =========================================================

def classify_intent_gpt(user_text, context_text=""):
    """
    text    = pesan user terbaru
    context = 5 chat sebelumnya
    """

    prompt = f"""
    Anda adalah sistem klasifikasi intent untuk chat pelanggan jasa pembuatan website.

    --- CONTEXT CHAT SEBELUMNYA ---
    {context_text}

    --- PESAN TERBARU ---
    USER: \"\"\"{user_text}\"\"\"

    Analisis maksud user berdasarkan:
    - isi pesan terbaru
    - keseluruhan context chat sebelumnya
    - topik yang sedang dibahas sebelumnya
    - tujuan yang tersirat, bukan hanya literal kalimat

    ------------------------------
    parent_intent (pilih salah satu):
    - perpanjang
    - tanya_status
    - minta_revisi
    - komplain
    - lainnya

    ------------------------------
    sub_intent:
    Jika parent_intent = perpanjang:
        - tanya_tagihan
        - tanya_masa_aktif
        - ingin_bayar
        - minta_invoice
        - konfirmasi_akan_perpanjang
        - kirim_bukti_bayar
        - atas_nama_bayar
        - tidak_perpanjang
        - konfirmasi_sukses_perpanjang

    Jika parent_intent = tanya_status:
        - status_pengerjaan
        - status_domain
        - status_update
        - status_perpanjangan
        - tanya_fasilitas

    Jika parent_intent = revisi:
        - revisi_konten
        - revisi_artikel
        - revisi_gambar
        - status_revisi
        - minta_akses_email

    Jika parent_intent = komplain:
        - komplain_harga
        - komplain_layanan
        - komplain_respon_lama
        - komplain_performa
        - komplain_hasil_revisi

    Jika parent_intent = lainnya:
        - salam
        - basa_basi
        - tidak_jelas

    ------------------------------
    Format jawaban HARUS seperti ini (tanpa tambahan apapun):

    parent_intent: <nama_intent>
    sub_intent: <nama_sub_intent>
    """

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip()

    # defauil output
    parent = "lainnya"
    child = "tidak_jelas"

    for line in res.split("\n"):
        s = line.lower().strip()
        if s.startswith("parent_intent:"):
            parent = s.replace("parent_intent:", "").strip()
        elif s.startswith("sub_intent:"):
            child = s.replace("sub_intent:", "").strip()

    return parent, child

# =========================================================
# SENTIMENT SCORING
# =========================================================

def analyze_sentiment(text, context):
    """
    Sentiment analysis berbasis GPT dengan skor 0â€“100 dan reasoning.
    Kontekstual: mempertimbangkan konteks sebelumnya bila diberikan.
    """

    # Build prompt kontekstual
    prompt = f"""
    Anda adalah model analisis sentiment tingkat lanjut.

    Tugas:
    1. Nilai *sentiment* dari teks user.
    2. Gunakan konteks percakapan "context" untuk menilai apakah emosi sebenarnya
        negatif, netral, atau positif.
    3. Output harus sangat konsisten dan numerik.

    Aturan penilaian (0-100):
    - 0â€“20   : Sangat Negatif â†’ kemarahan, kekecewaan berat, ancaman, hinaan, tekanan tinggi
    - 21â€“40  : Negatif         â†’ keluhan, frustrasi ringan-menengah, tidak puas
    - 41â€“60  : Netral          â†’ informasi, pertanyaan, penjelasan tanpa emosi
    - 61â€“80  : Positif         â†’ rasa terima kasih, puas, senang
    - 81â€“100 : Sangat Positif  â†’ antusiasme tinggi, bahagia, excitement besar
    Contoh negatif:
    - "Kok balasnya lama, waktu nagih aja cepat."
    - "Ini parah sekali, masa begini hasilnya?"
    - "Masih salah ini revisi ini belum sesuai"
    - "Kan saya udah bayar, kok masih ditagih?"

    Contoh positif:
    - "Makasih banyak kak"
    - "Bisa bantu SEO atau iklan kak?"
    - "Sip, sudah bagus dan sesuai"
    - "Saya mau pakai layanan lain bisa?"

    Contoh netral:
    - "Nanti ya"
    - "Bisa jelaskan ya kak"
    - "Baik kak saya diskusikan dulu"

    Sekarang analisis input berikut:
    ---
    Konsep utama / konteks: {context}
    Teks user: {text}
    ---

    Jawab HANYA dengan angka (0â€“100)
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    raw = response.output_text.strip()
    return extract_int(raw)

# =========================================================
# PRIORITY SCORING
# =========================================================

def compute_priority(intent_parent, intent_child, sentiment_score, text, context):

    prompt = f"""
    Anda adalah sistem penilai PRIORITY SCORE untuk perpanjangan website.

    Tugas Anda:
    Buat skor PRIORITY 0â€“100 berdasarkan:
    - intent_parent (kategori utama)
    - intent_child (sub intent yang lebih spesifik)
    - sentiment_score (0â€“100)
    - isi pesan user
    - konteks percakapan sebelumnya
    Prinsip penilaian PRIORITY:
    1. Permintaan terkait pembayaran, tagihan, invoice, konfirmasi pembayaran â†’ PRIORITY sangat tinggi (80â€“100)
    2. Revisi pada website setelah melakukan perpanjangan â†’ tinggi (70â€“79)
    3. User ingin perpanjang, menanyakan masa aktif, menjanjikan tanggal untuk perpanjang â†’ sedang (40â€“69)
    4. Komplain keras â†’ tinggi bila keluhan panjang, rendah bila hanya keluhan ringan (30â€“39)
    5. Tidak berminat atau menolak lanjut â†’ rendah (0â€“29)

    Gunakan kombinasi:
    - urgensi,
    - nilai bisnis,
    - kebutuhan respon cepat,
    - potensi kehilangan pelanggan,
    - probabilitas menghasilkan transaksi,
    - mood user (dari sentiment_score).
    Jawab HANYA dengan angka (0â€“100)

    Berikut datanya:
    ---
    intent_parent: {intent_parent}
    intent_child: {intent_child}
    sentiment_score: {sentiment_score}
    text: {text}
    context: {context}
    ---
    """
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    raw = response.output_text.strip()
    return extract_int(raw)

# =========================================================
# THREADED WORKER
# =========================================================

def process_row(idx, row):
    """
    This function processes one row completely. 
    It will be executed inside a thread.
    """
    try:
        context = build_gpt_context(row)

        intent_parent, intent_child = classify_intent_gpt(
            user_text=row["user_message"],
            context_text=context
        )

        sentiment = analyze_sentiment(
            row["user_message"],
            context
        )

        priority = compute_priority(
            intent_parent,
            intent_child,
            sentiment,
            row["user_message"],
            context
        )

        return {
            "index": idx,
            "intent_parent": intent_parent,
            "intent_child": intent_child,
            "sentiment": sentiment,
            "priority_score": priority
        }

    except Exception as e:
        print(f"[ERROR] Row {idx}: {e}")
        return {
            "index": idx,
            "intent_parent": "error",
            "intent_child": "error",
            "sentiment": 0,
            "priority_score": 0
        }

# =========================================================
# MAIN EXECUTION WITH THREAD POOL
# =========================================================

print("[LOAD] Loading filtered pairs...")
df = pd.read_json(PAIR_FILTERED_PATH)
print("[LOAD] Rows:", len(df))

results = []

print(f"[THREAD] Processing with max_workers={MAX_WORKERS}")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = [
        executor.submit(process_row, idx, row)
        for idx, row in df.iterrows()
    ]

    for future in as_completed(futures):
        results.append(future.result())

# =========================================================
# MERGE RESULTS BACK TO DATAFRAME
# =========================================================

result_df = pd.DataFrame(results).set_index("index")
result_df = result_df.sort_index()

df["intent_parent"] = result_df["intent_parent"]
df["intent_child"] = result_df["intent_child"]
df["sentiment"] = result_df["sentiment"]
df["priority_score"] = result_df["priority_score"]

# =========================================================
# SAVE FINAL RESULT
# =========================================================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print("[DONE] Saved:", OUTPUT_PATH)
