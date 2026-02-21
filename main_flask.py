import json
import os
from dotenv import load_dotenv
load_dotenv()
import uuid
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

from db import (
    insert_chat_pair,
    fetch_context,
    fetch_dataset_by_intent
)
from db import insert_chat_pair, fetch_next_turn_index
from db import apply_feedback_db

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY belum diset.")

_client_warmup = OpenAI(api_key=OPENAI_API_KEY)
_client_warmup.chat
_client_warmup.embeddings

import logging

# ============================================================
# LOGGING SETUP
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/chatbot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("perpanjangan-chatbot")

# ============================================================
# CONFIG 
# ============================================================
CLIENT = OpenAI(api_key=OPENAI_API_KEY)
TOP_K = 3

EXECUTOR = ThreadPoolExecutor(max_workers=2,
                              thread_name_prefix="openai-worker-")

# ============================================================
# PLACEHOLDERS
# ============================================================
REQUIRED_PLACEHOLDERS = {
    "tanya_tagihan": ["{{$biaya_ppj_web}}"],
    "tanya_masa_aktif": ["{{$jatuh_tempo}}", "{{$domain_klien}}"],
    "status_perpanjangan": ["{{$jatuh_tempo}}", "{{$domain_klien}}"],
    "status_domain": ["{{$domain_klien}}"],
    "minta_invoice": ["{{$domain_klien}}"]
}

# ============================================================
# LOAD DATASET
# ============================================================
def clean_bot_output(text):  #bersihin data 
    text = re.sub(r"[ð]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_user_query(text):
    if not text:
        return text

    # Ambil pesan user terakhir saja
    parts = re.split(r"\buser\s*:\b", text, flags=re.I)
    last_user = parts[-1]

    # Buang bagian assistant jika ada
    last_user = re.split(r"\bassistant\s*:\b", last_user, flags=re.I)[0]

    return last_user.strip()

# ============================================================
# EMBEDDING & INTENT
# ============================================================

def generate_embedding(text):
    res = CLIENT.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return res.data[0].embedding

def classify_intent_gpt(user_text, context):
    """
    text    = pesan user terbaru
    context = 5 chat sebelumnya
    """

    prompt = f"""
    Anda adalah sistem klasifikasi intent untuk chat pelanggan jasa pembuatan website.

    --- CONTEXT CHAT SEBELUMNYA ---
    {context}

    --- PESAN TERBARU ---
    USER: \"\"\"{user_text}\"\"\"

    Analisis maksud user berdasarkan:
    - isi pesan terbaru
    - keseluruhan context chat sebelumnya
    - topik yang sedang dibahas sebelumnya
    - tujuan yang tersirat, bukan hanya literal kalimat

    ------------------------------
    inferred_parent (pilih salah satu):
    - perpanjang
    - tanya_status
    - minta_revisi
    - komplain
    - lainnya

    ------------------------------
    inferred_child:
    Jika inferred_parent = perpanjang:
        - tanya_tagihan
        - tanya_masa_aktif
        - ingin_bayar
        - minta_invoice
        - konfirmasi_akan_perpanjang
        - kirim_bukti_bayar
        - atas_nama_bayar
        - tidak_perpanjang
        - konfirmasi_sukses_perpanjang

    Jika inferred_parent = tanya_status:
        - status_pengerjaan
        - status_domain
        - status_update
        - status_perpanjangan
        - tanya_fasilitas
        - tanya_domain

    Jika inferred_parent = revisi:
        - revisi_konten
        - revisi_artikel
        - revisi_gambar
        - status_revisi
        - minta_akses_email

    Jika inferred_parent = komplain:
        - komplain_harga
        - komplain_layanan
        - komplain_respon_lama
        - komplain_performa
        - komplain_hasil_revisi

    Jika inferred_parent = lainnya:
        - salam
        - basa_basi
        - tidak_jelas

    ------------------------------
    Format jawaban HARUS seperti ini (tanpa tambahan apapun):

    inferred_parent: <nama_intent>
    inferred_child: <nama_sub_intent>
    """

    res = CLIENT.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip()

    # default output
    inferred_parent = "lainnya"
    inferred_child = "tidak_jelas"

    for line in res.split("\n"):
        s = line.lower().strip()
        if s.startswith("inferred_parent:"):
            inferred_parent = s.replace("inferred_parent:", "").strip()
        elif s.startswith("inferred_child:"):
            inferred_child = s.replace("inferred_child:", "").strip()
    return inferred_parent, inferred_child

# ============================================================
# THREADPOOL WRAPPERS
# ============================================================

def generate_embedding_async(text):
    return EXECUTOR.submit(generate_embedding, text)

def classify_intent_async(user_text, context):
    return EXECUTOR.submit(classify_intent_gpt, user_text, context)

# ============================================================
# COSINE SIMILARITY & TOP K RETRIEVAL
# ============================================================

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_k(query_embedding, df, k=3):
    # === COPY EMBEDDING ===
    df_temp = df.copy()

    # === COSINE SIMILARITY ===
    df_temp["similarity"] = df_temp["embedding"].apply(
        lambda e: cosine_sim(e, query_embedding)
    )

    # === NORMALISASI ===
    sim_min = df_temp["similarity"].min()
    sim_max = df_temp["similarity"].max()
    df_temp["similarity_norm"] = (
    (df_temp["similarity"] - sim_min) /
    (sim_max - sim_min + 1e-6)
    )

    df_temp["similarity_norm_100"] = df_temp["similarity_norm"] * 100

    # === FINAL SCORE ===
    df_temp["final_score"] = (
    df_temp["similarity_norm_100"] * 0.7 +
    df_temp["priority_score"] * 0.3
    ).round().astype(int)

    return df_temp.sort_values("final_score", ascending=False).head(k)

# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt_from_matches(user_text, matches_df):

    sections = []

    for _, row in matches_df.iterrows():
        ctx_text = "\n".join(row["context"])
        adm = row["admin_response"]
        usr = row["user_message"]

        section = f"""
    === MATCH ===
    Context:
    {ctx_text}

    User says:
    {usr}

    Admin replied:
    {adm}
    """
        sections.append(section)

    match_block = "\n\n".join(sections)

    prompt = f"""
    Anda adalah AI Customer Service untuk layanan Perpanjangan Website.

    PLACEHOLDER MODE (WAJIB):

    Jika user menanyakan:
    - biaya perpanjangan
    - jatuh tempo
    - masa aktif
    - nama website
    - pembayaran / invoice
    
    Gunakan placeholder variabel berikut JIKA DAN HANYA JIKA relevan:
    - Nama website: {{$domain_klien}}
    - Jatuh tempo perpanjangan: {{$jatuh_tempo}}
    - Biaya perpanjangan: {{$biaya_ppj_web}}

    ATURAN STRUKTUR PLACEHOLDER (WAJIB):

    - Placeholder {{...}} adalah NILAI FINAL, bukan kata benda atau objek kalimat
    - Placeholder HARUS muncul sebagai:
    - akhir kalimat, ATAU
    - setelah tanda ":" ATAU
    - setelah kata kerja langsung tanpa preposisi

    ATURAN MUTLAK (WAJIB DIPATUHI):
    - Jika user menanyakan biaya, jatuh tempo, atau nama website:
      WAJIB gunakan placeholder variabel {{...}} persis seperti tertulis.
    - DILARANG mengganti placeholder dengan nilai contoh dari database.
    - DILARANG mengira-ngira.
    - Jika melanggar aturan ini, jawaban dianggap SALAH.
    - Mengarahkan user ke pembuatan invoice TIDAK BOLEH dilakukan jika user secara langsung meminta nomor rekening

    CONTOH BENAR:
    - "Jatuh tempo perpanjangan: {{$jatuh_tempo}}"
    - "Masa aktif website berlaku sampai {{$jatuh_tempo}}"

    CONTOH SALAH (DILARANG):
    - "informasi {{$jatuh_tempo}}"
    - "detail {{$biaya_ppj_web}}"

    FORMAT JAWABAN:
    - Gunakan bahasa profesional dan ramah
    - Placeholder {{...}} HARUS DITULIS UTUH, TIDAK BOLEH DIMODIFIKASI
    - Jangan menambahkan angka atau tanggal selain placeholder

    PRIORITAS JAWABAN:
    1. Jika user bertanya "nomor rekening", "rekening pembayaran", "transfer ke mana","bayar ke mana"
	    - Jawab informasi rekening sebagai berikut:
		"Kami tunggu konfirmasi perpanjangannya, bisa ditransfer ke rekening berikut : 
         BCA    : 0990911185
         BRI    : 066101002265301
         An. CV Eksa Digital Marketing"
    2. Invoice hanya disebut jika:
        - user meminta invoice, ATAU
        - user meminta tagihan resmi tertulis
    
    LAYANAN GRATIS PERPANJANGAN WEBSITE:
    1. Fasilitas Penunjang Perpanjangan Website
        - Penambahan 1 artikel dan 5 kata kunci pencarian yang dapat diminta.
        - Penambahan 10 foto bisnis terbaru.
        - Pengeditan sederhana di website seperti mengganti Nomor telepon/WA, dan Alamat usaha (jika pindah alamat)
        - Desain Bisnis gratis : banner/kartu nama/logo (pilih salah satu) dapat dikirim dalam file PSD.
    2. Selalu tekankan layanan GRATIS ini dalam jawaban Anda jika relevan, jika klien menanyakan diluar layanan gratis, maka jawab dengan sopan bahwa layanan tersebut di luar layanan gratis dan ada tambahan biaya, kemudian izin untuk menginformasikan ke tim terkait.

    Berikut adalah {len(matches_df)} percakapan paling mirip dari database:

    {match_block}

    Tugas Anda:
    - Berikan jawaban final profesional dan sopan.
    - Gunakan gaya admin dari contoh-contoh di atas.
    - Jawaban harus relevan dengan pertanyaan user.
    - Jangan menambah informasi palsu.
    - Hindari memberikan jawaban berkaitan dengan perpanjangan yang belum ada di database, lebih baik menjawab akan menanyakan pada tim terkait.
    - Jika tidak yakin, minta detail tambahan dari user.
    - Gunakan baris baru (newline) jika diperlukan agar jawaban mudah dibaca.
    - Gunakan bahasa ramah dan profesional
    - Gunakan nada hangat dan empati, boleh sesekali memanggil nama user jika sudah perkenalan, namun jangan selalu memanggil nama user di setiap jawaban.
    - Jangan memberikan jawaban yang berbelit, terlalu singkat atau terlalu panjang.
    - Gunakan emoticon yang relevan dan tidak repetitive untuk meningkatkan kehangatan dalam komunikasi.
    - Berikan jawaban nomor rekening yang pada dataset mengandung kalimat CV Eksa Digital Marketing

    USER QUERY:
    "{user_text}"
    """

    return prompt

# ============================================================
# PLACEHOLDER GUARD
# ============================================================

def enforce_placeholders(user_text, draft_text, inferred_child):
    required = REQUIRED_PLACEHOLDERS.get(inferred_child)
    if not required:
        return draft_text

    missing = [p for p in required if p not in draft_text]

    if not missing:
        return draft_text

    guard_prompt = f"""
    Anda adalah VALIDATOR dan EDITOR jawaban AI.

    PERAN ANDA:
    - BUKAN membuat jawaban baru
    - BUKAN mengubah gaya bahasa utama
    - TIDAK menambahkan informasi baru
    - TIDAK menghapus maksud jawaban

    TUGAS UTAMA:
    Memastikan jawaban MEMATUHI kontrak PLACEHOLDER.

    ====================================
    DATA
    ====================================

    PESAN USER:
    {user_text}

    JAWABAN DRAFT:
    {draft_text}

    PLACEHOLDER WAJIB:
    {", ".join(required)}

    ATURAN DATA NUMERIK (SANGAT PENTING):

    1. DATA YANG WAJIB DILINDUNGI DENGAN PLACEHOLDER:
        - Jatuh tempo / tanggal aktif
        - Biaya perpanjangan individual
        - Nama domain klien
        - Tagihan spesifik klien

        ➜ Data ini TIDAK BOLEH muncul sebagai angka, tanggal, atau teks nyata.
        ➜ WAJIB menggunakan placeholder {{...}} jika relevan.

    2. DATA YANG BOLEH DITULIS SECARA LITERAL:
        - Kode layanan atau paket
        - Informasi pembayaran umum
        - Nomor customer service
        - Informasi operasional non-klien
        ➜ Data ini BOLEH ditampilkan apa adanya jika sudah ada di draft.

    3. Jika menemukan angka:
        - Periksa konteks kalimatnya
        - Jika angka terkait DATA KLIEN → GUNAKAN PLACEHOLDER
        - Jika angka terkait DATA UMUM → BIARKAN

    4. DILARANG:
        - Menghapus nomor rekening yang sudah benar
        - Mengganti nomor rekening dengan placeholder
        - Menyensor data pembayaran umum

    5. Informasi bank pembayaran:
        BCA    : 0990911185
        BRI    : 066101002265301
        An. CV Eksa Digital Marketing

    ====================================
    ATURAN PLACEHOLDER (WAJIB)
    =====================================

    1. Jika placeholder WAJIB belum muncul:
    - Tambahkan placeholder tersebut secara natural
    - Gunakan struktur kalimat yang sederhana

    2. Placeholder adalah NILAI FINAL:
        - BUKAN objek
        - BUKAN keterangan tambahan

    ✔ BOLEH:
    - "Biaya perpanjangan: {{$biaya_ppj_web}}"
    - "Masa aktif website berlaku sampai {{$jatuh_tempo}}"

    ✘ DILARANG:
    - "informasi {{$biaya_ppj_web}}"
    - "detail {{$domain_klien}}"

    3. Placeholder TIDAK BOLEH:
    - didahului kata: pada, di, tentang, seputar, informasi, detail, yaitu
    - berada di tengah klausa yang panjang

    4. Placeholder HARUS:
    - di akhir kalimat ATAU
    - setelah tanda ":" ATAU
    - setelah kata kerja langsung

    PENGECUALIAN PENTING:
    - Jika USER secara eksplisit meminta NOMOR REKENING atau INFO PEMBAYARAN UMUM
    - DAN data tersebut adalah DATA UMUM (bukan spesifik klien)
    - MAKA Anda BOLEH menambahkan informasi tersebut meskipun tidak ada di draft

    ====================================
    ATURAN MUTLAK
    ====================================

    - DILARANG mengira-ngira data
    - DILARANG menambah topik baru
    - DILARANG menghapus informasi penting dari draft

    Jika draft sudah BENAR:
    - Kembalikan draft apa adanya (tanpa perubahan)

    ====================================
    OUTPUT
    ====================================

    Kembalikan HANYA jawaban final.
    Tanpa penjelasan.
    Tanpa catatan.
    Tanpa format tambahan.
    Pastikan jawaban sesuai dengan pertanyaan user dan tidak berbelit.
    """

    res = CLIENT.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": guard_prompt}]
    )

    return clean_bot_output(res.choices[0].message.content)

# ============================================================
# GENERATE BOT RESPONSE 
# ============================================================
MAX_CONTEXT_TURNS = 6

def generate_bot_reply_with_context(user_text, context_text, matches_df):
    prompt_matches = build_prompt_from_matches(user_text, matches_df)

    final_prompt = f"""
    === RIWAYAT PERCAKAPAN SEBELUMNYA ===
    {context_text}

    === REFERENSI JAWABAN DARI DATABASE ===
    {prompt_matches}

    === PESAN TERBARU DARI USER ===
    "{user_text}"

    Berikan jawaban yang:
    - Konsisten dengan konteks percakapan sebelumnya
    - Tidak mengulang pertanyaan yang sudah ada dijawab
    - Tidak menyangkal informasi yang sudah diberikan
    """

    res = CLIENT.chat.completions.create(
        model="gpt-4.1-mini",
                messages=[
            {"role": "system", "content": "Anda adalah AI admin pelayanan perpanjangan website."},
            {"role": "user", "content": final_prompt}
        ]
    )
    return clean_bot_output(res.choices[0].message.content.strip())

def save_chat_to_db(
    conversation_id: int,
    user_message: str,
    admin_response: str,
    intent_parent: str = None,
    intent_child: str = None,
    priority_score: int = 50,
    embedding: list = None,
    context: list = None,
    session_id: str = None
):
    if session_id is None:
        session_id = str(uuid.uuid4())

    turn_index = fetch_next_turn_index(conversation_id)

    insert_chat_pair({
        "conversation_id": conversation_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "user_message": user_message,
        "admin_response": admin_response,
        "context": context or [],
        "intent_parent": intent_parent,
        "intent_child": intent_child,
        "priority_score": priority_score,
        "reward_count": 0,
        "punish_count": 0,
        "embedding": embedding or []
    })

# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Perpanjangan chatbot API running"
    })

# ============================================================
# ENDPOINT /chat
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_query = payload.get("query") or payload.get("q")
    user_query = normalize_user_query(user_query)
    conversation_id = payload.get("conversation_id")

    if conversation_id:
        conversation_id = int(conversation_id)
    else:
        conversation_id = int(uuid.uuid4().int % 1_000_000_000)

    if not user_query:
        return jsonify({"error": "query required"}), 400
    
    logger.info(
        f"[REQUEST] conversation_id={conversation_id} | query='{user_query}'"
    )
    if not user_query:
        logger.warning(
            f"[INVALID REQUEST] conversation_id={conversation_id} | empty query"
        )
        return jsonify({"error": "query required"}), 400

    # === CONTEXT ===
    context_list = fetch_context(conversation_id)
    context_text = "\n".join(context_list)
    logger.debug(
        f"[CONTEXT] conversation_id={conversation_id} | turns={len(context_list)}"
    )
    session_id = str(uuid.uuid4())
    logger.info(
        f"[SESSION] conversation_id={conversation_id} | session_id={session_id}"
    )

    # === PARALLEL EXECUTION ===
    intent_future = classify_intent_async(user_query, context_text)
    embedding_future = generate_embedding_async(user_query)

    # === TUNGGU HASILNYA ===
    query_embedding = embedding_future.result()
    
    # === GET INTENT RESULT ===
    try:
        inferred_parent, inferred_child = intent_future.result()
    except Exception as e:
        logger.error(
            f"[INTENT ERROR] session_id={session_id} | {str(e)}",
            exc_info=True
    )
        inferred_parent, inferred_child = "lainnya", "tidak_jelas"
    
    logger.info(
        f"[INTENT] session_id={session_id} | parent={inferred_parent} | child={inferred_child}"
    )

    logger.debug(
        f"[EMBEDDING] session_id={session_id} | vector_dim={len(query_embedding)}"
    )
    # === RETRIEVAL (DATA LAMA) ===
    dataset = fetch_dataset_by_intent(
        inferred_parent if inferred_parent != "lainnya" else None
    )
    df_retrieval = pd.DataFrame(dataset)
    logger.info(
        f"[RETRIEVAL] intent_parent={inferred_parent} | candidates={len(df_retrieval)}"
    )
    if df_retrieval.empty:
        bot_text = "Baik kak, untuk hal ini kami perlu cek dulu ke tim terkait ya 🙏"

        turn_index = fetch_next_turn_index(conversation_id)

        insert_chat_pair({
            "conversation_id": conversation_id,
            "session_id": session_id,
            "turn_index": turn_index,
            "user_message": user_query,
            "admin_response": bot_text,
            "context": context_list,
            "intent_parent": inferred_parent,
            "intent_child": inferred_child,
            "priority_score": 50,
            "reward_count": 0,
            "punish_count": 0,
            "embedding": query_embedding
        })

        return jsonify({
            "status": "ok",
            "admin_response": bot_text
        })

    matches_df = retrieve_top_k(query_embedding, df_retrieval, TOP_K)
    if not matches_df.empty:
        top = matches_df.iloc[0]
        logger.info(
            f"[TOP MATCH] sim={top['similarity']:.4f} | "
            f"priority={top['priority_score']} | "
            f"final={top['final_score']}"
        )
    else:
        logger.warning(
            f"[RETRIEVAL EMPTY] session_id={session_id}"
        )
    if matches_df is None or len(matches_df) == 0:
        top_similarity = 0.0
        top_match = None
    else:
        top_match = matches_df.iloc[0]
        top_similarity = float(top_match["similarity"])

    matches_summary = []
    if matches_df is not None:
        for _, r in matches_df.iterrows():
            matches_summary.append({
                "conversation_id": int(r["conversation_id"]) if not pd.isna(r["conversation_id"]) else None,
                "intent_parent": r.get("intent_parent"),
                "intent_child": r.get("intent_child"),
                "user_message": r.get("user_message"),
                "admin_response": r.get("admin_response"),
                "similarity" : float(r.get("similarity") or 0.0),
                "priority_score": float(r.get("priority_score") or 0.0),
                "final_score": float(r.get("final_score") or 0.0)
            })
 
    # =====================================================
    # GENERATE RESPONSE
    # =====================================================

    draft_text = generate_bot_reply_with_context(
        user_query, context_text, matches_df
    )
    logger.info(
        f"[LAYER-1 DRAFT] session_id={session_id} | "
        f"answer={draft_text}"
    )
    logger.debug(
        f"[LAYER-1 CONTENT]\n{draft_text}"
    )

    bot_text = enforce_placeholders(
        user_query,
        draft_text,
        inferred_child
    )
    logger.info(
        f"[LAYER-2 FINAL] session_id={session_id} | "
        f"answer={bot_text}"
    )

    if bot_text != draft_text:
        logger.warning(
            f"[PLACEHOLDER GUARD APPLIED] session_id={session_id}"
        )

    logger.debug(
        f"[FINAL CONTENT]\n{bot_text}"
    )

    # =====================================================
    # GENERATE & SAVE
    # =====================================================

    turn_index = fetch_next_turn_index(conversation_id)

    new_row = {
        "conversation_id": conversation_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "user_message": user_query,
        "admin_response": bot_text,
        "context": context_list,
        "intent_parent": inferred_parent,
        "intent_child": inferred_child,
        "priority_score": 50,
        "reward_count": 0,
        "punish_count": 0,
        "embedding": query_embedding
    }

    logger.info(
        f"[DB INSERT] conversation_id={conversation_id} | "
        f"turn_index={turn_index} | session_id={session_id}"
    )
    try:
        insert_chat_pair(new_row)
    except Exception as e:
        logger.critical(
            f"[DB ERROR] session_id={session_id} | {str(e)}",
            exc_info=True
        )
        raise

    return jsonify({
        "status": "ok",
        "save_mode": "saved_to_dataset",
        "session_id": session_id,
        "user_message": user_query,
        "admin_response": bot_text,
        "intent_parent": inferred_parent,
        "intent_child": inferred_child,
        "matches": matches_summary
    })

# ============================================================
# ENDPOINT /feedback
# ============================================================

@app.route("/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True) or {}

    session_id = payload.get("session_id")
    rating = payload.get("rating")
    logger.info(
        f"[FEEDBACK] session_id={session_id} | rating={rating}"
    )

    if not session_id or rating not in [-1, 0, 1]:
        return jsonify({"error": "invalid input"}), 400

    if rating == 0:
        return jsonify({
            "status": "ok",
            "message": "no feedback applied"
        })

    success = apply_feedback_db(session_id, rating)

    if not success:
        return jsonify({
            "error": "session_id not found"
        }), 404

    return jsonify({
        "status": "ok",
        "session_id": session_id,
        "rating": rating,
        "message": "feedback saved to sqlite"
    })

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
