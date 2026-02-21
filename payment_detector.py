# from openai import OpenAI
# from datetime import datetime
# import json

# CLIENT = OpenAI(api_key="REDACTED_API_KEY")

# def detect_payment(message_text=None, image_url=None):
#     """
#     Return dict:
#     {
#       is_payment: bool,
#       confidence: float,
#       extracted_name: str | None,
#       extracted_amount: str | None,
#       extracted_bank: str | None
#     }
#     """

#     prompt = f"""
# Anda adalah SISTEM DETEKSI BUKTI PEMBAYARAN (BACKEND VALIDATOR).

# PRINSIP MUTLAK:
# - JANGAN mengarang data
# - JANGAN menyimpulkan jika bukti tidak jelas
# - Jika ragu â†’ is_payment = false
# - Confidence harus jujur

# KRITERIA BUKTI BAYAR SAH:
# - Screenshot transfer bank
# - Bukti QRIS
# - Invoice berstatus PAID
# - Mutasi rekening
# - Kalimat eksplisit:
#   "sudah saya transfer"
#   "ini bukti pembayarannya"
#   "pembayaran sudah dilakukan"

# DILARANG:
# - Menganggap chat biasa sebagai pembayaran
# - Mengisi field jika tidak ada bukti eksplisit

# OUTPUT HARUS JSON VALID TANPA TEKS TAMBAHAN:

# {{
#   "is_payment": true | false,
#   "confidence": 0.0,
#   "extracted_name": null,
#   "extracted_amount": null,
#   "extracted_bank": null
# }}

# DATA YANG DITERIMA:
# TEXT:
# {message_text}

# IMAGE_URL (jika ada):
# {image_url}
# """

#     try:
#         res = CLIENT.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )

#         raw = res.choices[0].message.content.strip()
#         data = json.loads(raw)

#         # HARD GUARD
#         return {
#             "is_payment": bool(data.get("is_payment", False)),
#             "confidence": float(data.get("confidence", 0.0)),
#             "extracted_name": data.get("extracted_name"),
#             "extracted_amount": data.get("extracted_amount"),
#             "extracted_bank": data.get("extracted_bank")
#         }

#     except Exception as e:
#         # FAIL SAFE
#         return {
#             "is_payment": False,
#             "confidence": 0.0,
#             "extracted_name": None,
#             "extracted_amount": None,
#             "extracted_bank": None
#         }

