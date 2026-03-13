# import json
# import os
# import re
# import time
# from pathlib import Path
# from urllib.parse import quote
#
# URL_REGEX = re.compile(r"https?://[^\s'\"<>()]+", flags=re.I)
#
#
# def extract_urls_from_text(text):
#     if not text:
#         return []
#     urls = URL_REGEX.findall(str(text))
#     seen = set()
#     out = []
#     for u in urls:
#         if u and u not in seen:
#             seen.add(u)
#             out.append(u)
#     return out
#
#
# def _load_cache(cache_path):
#     p = Path(cache_path)
#     if not p.exists():
#         return {}
#     try:
#         return json.loads(p.read_text(encoding="utf-8"))
#     except Exception:
#         return {}
#
#
# def _save_cache(cache_path, cache):
#     p = Path(cache_path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
#
#
# def _strict_prompt():
#     return (
#         "You are a highly strict and structure-aware financial image validator.\n"
#         "Your job is ONLY to identify real payment receipts or genuine money transfer proofs.\n"
#         "================================================================================\n"
#         "REJECT the image (is_receipt=false) if it is:\n"
#         "- any advertisement, promotion, poster, banner, marketing design\n"
#         "- images showing big numbers like '4 JT', '5.000.000', '50%', etc. without transaction context\n"
#         "- pictures of people, models, products, or call center numbers\n"
#         "- screenshots of websites, catalogs, WhatsApp chats, or social media posts\n"
#         "- edited images, aesthetic graphics, or anything with handwriting\n"
#         "================================================================================\n"
#         "ONLY classify an image as a REAL receipt if BOTH conditions are satisfied:\n"
#         "(1) TEXT-BASED RECEIPT ELEMENTS (should appear clearly):\n"
#         "    - sender or payer information (name/ID/phone)\n"
#         "    - receiver or merchant name\n"
#         "    - bank/platform (BCA, BRI, Mandiri, QRIS, Dana, OVO, ShopeePay, Seabank, etc.)\n"
#         "    - transaction date and time\n"
#         "    - reference number / transaction ID / authorization code\n"
#         "    - transaction amount (Total / Amount Paid / Jumlah / Nominal)\n"
#         "    - payment method (transfer, QRIS, VA, debit, mobile banking)\n"
#         "If multiple of these are missing -> NOT a receipt.\n"
#         "(2) STRUCTURAL VISUAL FEATURES (detect at least TWO):\n"
#         "    - printed/machine-generated font (NOT handwritten)\n"
#         "    - structured receipt layout (header area, aligned rows, spacing)\n"
#         "    - merchant/bank logo in a header position\n"
#         "    - QR code or barcode block\n"
#         "    - consistent banking-app UI elements (uniform typography, aligned sections)\n"
#         "If structural features do NOT resemble real receipts -> NOT a receipt.\n"
#         "================================================================================\n"
#         "ADDITIONAL RULES:\n"
#         "- Fake receipts or manually designed templates must be rejected.\n"
#         "- A photo containing only a large number must be rejected.\n"
#         "- If uncertain, always classify as NOT a receipt.\n"
#         "RETURN RULE:\n"
#         "Return is_receipt=true ONLY when BOTH textual and structural criteria match real receipts.\n"
#         "Otherwise ALWAYS return is_receipt=false and payment_value=null.\n"
#         "Extract only the actual total amount paid by the customer (exclude admin fees if visible)."
#     )
#
#
# def _tool_schema():
#     return [
#         {
#             "type": "function",
#             "function": {
#                 "name": "receipt_decision",
#                 "description": "Strict receipt decision",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "is_receipt": {"type": "boolean"},
#                         "payment_value": {"type": ["number", "null"]},
#                         "reason": {"type": "string"},
#                     },
#                     "required": ["is_receipt", "payment_value", "reason"],
#                     "additionalProperties": False,
#                 },
#             },
#         }
#     ]
#
#
# def detect_receipt_url(client, url, cache, model):
#     if not url:
#         return {"is_receipt": False, "payment_value": None, "reason": "empty url", "url": url}
#     if url in cache:
#         cached = cache[url]
#         cached["url"] = url
#         return cached
#     if client is None:
#         res = {"is_receipt": False, "payment_value": None, "reason": "no api key", "url": url}
#         cache[url] = res
#         return res
#
#     encoded_url = quote(url.strip(), safe=":/?&=#.%")
#     for attempt in range(3):
#         try:
#             resp = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": _strict_prompt()},
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": "Analyze this payment receipt image."},
#                             {"type": "image_url", "image_url": {"url": encoded_url}},
#                         ],
#                     },
#                 ],
#                 tools=_tool_schema(),
#                 tool_choice="auto",
#                 temperature=0,
#             )
#             msg = resp.choices[0].message
#             tool_calls = getattr(msg, "tool_calls", None)
#             if not tool_calls:
#                 res = {"is_receipt": False, "payment_value": None, "reason": "no tool output", "url": url}
#                 cache[url] = res
#                 return res
#             parsed = json.loads(tool_calls[0].function.arguments)
#             res = {
#                 "is_receipt": bool(parsed.get("is_receipt")),
#                 "payment_value": parsed.get("payment_value"),
#                 "reason": str(parsed.get("reason", "")),
#                 "url": url,
#             }
#             cache[url] = res
#             return res
#         except Exception:
#             if attempt == 2:
#                 res = {"is_receipt": False, "payment_value": None, "reason": "api error", "url": url}
#                 cache[url] = res
#                 return res
#             time.sleep(1)
#
#
# def detect_receipt_urls(client, urls, cache_path):
#     model = os.getenv("PAYMENT_VISION_MODEL", "gpt-4o-mini")
#     cache = _load_cache(cache_path)
#     results = []
#     for url in urls:
#         results.append(detect_receipt_url(client, url, cache, model))
#     _save_cache(cache_path, cache)
#     return results, model
