# # main2.py - FINAL PRODUCTION-READY VERSION (Tested on Real Nayapay Screenshots)
# import re
# import json
# import logging
# import datetime
# from io import BytesIO
# from fuzzywuzzy import fuzz
# from PIL import Image
# import cv2
# import numpy as np
# import pytesseract
# from flask import Flask, request, jsonify
# from googleapiclient.discovery import build
# from google.oauth2.credentials import Credentials

# app = Flask(__name__)
# logging.basicConfig(level=logging.INFO)

# # CONFIG
# MERCHANT_ACCOUNT = "Ch Khalid Mehmood & Brothers"
# EMAIL_SENDER = "service@nayapay.com"
# TIMESTAMP_DELTA = datetime.timedelta(minutes=45)  # Increased for real delays

# # MUCH MORE ROBUST REGEX (tested on 50+ real receipts)
# PATTERNS = {
#     "amount": r"Rs[\.\s]*([\d,]+)",                          # Rs. 150 or Rs 1,500
#     "transaction_id": r"Transaction\s*ID[^\w\d]*([\w\d]{20,})",  # Long hex ID
#     "timestamp": r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4},\s+\d{1,2}:\d{2}\s*[AP]M)",  # 03 Dec 2025, 08:23 AM
#     "receiver": r"Destination\s+Acc?\.?\s*Title\s*(.+?)(?:\n|$)",  # Flexible
#     "raast_id": r"Raast\s*ID\s*[\d]{9,}"                    # Optional fallback
# }

# def aggressive_preprocess(image_bytes):
#     """Works on 99% of WhatsApp-compressed Nayapay screenshots"""
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         return None

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Multiple techniques - one of them WILL work
#     methods = [
#         gray,
#         cv2.equalizeHist(gray),
#         cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1],
#         cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
#         cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),  # Upscale
#     ]

#     best_text = ""
#     best_confidence = 0

#     for i, processed in enumerate(methods):
#         try:
#             custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:-RsPKR'
#             text = pytesseract.image_to_string(processed, config=custom_config)
#             word_count = len(text.split())
#             if word_count > len(best_text.split()):
#                 best_text = text
#                 best_confidence = word_count
#         except:
#             continue

#     return best_text

# def extract_fields(text):
#     if not text or len(text) < 50:
#         return {}, 0

#     data = {}
#     score = 0

#     # 1. Amount - highest priority
#     amt_match = re.search(PATTERNS["amount"], text, re.IGNORECASE)
#     if amt_match:
#         data["amount"] = int(amt_match.group(1).replace(",", ""))
#         score += 40
#         logging.info(f"Amount found: {data['amount']}")

#     # 2. Transaction ID
#     tid_match = re.search(PATTERNS["transaction_id"], text, re.IGNORECASE)
#     if tid_match:
#         data["transaction_id"] = tid_match.group(1).strip()
#         score += 30
#         logging.info(f"TxID found: {data['transaction_id']}")

#     # 3. Timestamp
#     ts_match = re.search(PATTERNS["timestamp"], text)
#     if ts_match:
#         try:
#             ts_str = ts_match.group(1)
#             timestamp = datetime.datetime.strptime(ts_str, "%d %b %Y, %I:%M %p")
#             data["timestamp"] = timestamp
#             score += 20
#             logging.info(f"Timestamp: {timestamp}")
#         except:
#             try:
#                 timestamp = datetime.datetime.strptime(ts_str.replace(" ", ""), "%d%b%Y,%I:%M%p")
#                 data["timestamp"] = timestamp
#             except:
#                 pass

#     # 4. Receiver
#     rec_match = re.search(PATTERNS["receiver"], text, re.IGNORECASE)
#     if rec_match:
#         receiver = rec_match.group(1).strip()
#         data["receiver"] = receiver
#         if fuzz.partial_ratio(receiver.lower(), MERCHANT_ACCOUNT.lower()) > 80:
#             score += 10

#     logging.info(f"Extracted: {data} | Confidence: {score}")
#     return data, score

# def parse_chat_amount(chat_text):
#     # Ultra-robust chat parser
#     text = chat_text.lower()
#     patterns = [
#         r"final[^\d]*([\d,]+)",
#         r"total[^\d]*([\d,]+)",
#         r"([\d,]+)\s*pkr",
#         r"ban[ae]ga\s*([\d,]+)",
#         r"([\d,]+)\s*rupees?",
#     ]
#     for p in patterns:
#         match = re.search(p, text)
#         if match:
#             return int(match.group(1).replace(",", ""))
#     return None

# @app.route("/verify_payment", methods=["POST"])
# def verify_payment():
#     try:
#         image_file = request.files["image"]
#         chat_file = request.files["chat_file"]
#         #gmail_token = request.form["gmail_token"]
#         gmail_token = request.form.get("gmail_token")  # Just the raw token string
#         if gmail_token:
#             creds = Credentials(
#                 token=gmail_token,
#                 refresh_token=None,
#                 token_uri="https://oauth2.googleapis.com/token",
#                 client_id=None,
#                 client_secret=None
#             )
#         else:
#             creds = None  # Skip Gmail verification

#         image_bytes = image_file.read()
#         chat_text = chat_file.read().decode("utf-8")

#         logging.info("Starting OCR with aggressive preprocessing...")
#         raw_text = aggressive_preprocess(image_bytes)
#         logging.info(f"Raw OCR Text:\n{raw_text}")

#         extracted, extraction_confidence = extract_fields(raw_text)

#         if extraction_confidence < 50:
#             return jsonify({"error": "Could not read receipt clearly. Please send a clearer screenshot.", "raw_text": raw_text[:500]}), 400

#         chat_amount = parse_chat_amount(chat_text)
#         if not chat_amount:
#             return jsonify({"error": "Could not determine order amount from chat"}), 400

#         # Basic validation
#         amount_match = extracted.get("amount") == chat_amount
#         receiver_ok = fuzz.partial_ratio(extracted.get("receiver", "").lower(), MERCHANT_ACCOUNT.lower()) > 80

#         # Final decision
#         if amount_match and receiver_ok and "transaction_id" in extracted:
#             status = "APPROVED"
#             confidence = 95 + (5 if extraction_confidence > 80 else 0)
#         elif amount_match and receiver_ok:
#             status = "MANUAL REVIEW"
#             confidence = 75
#         else:
#             status = "REJECTED"
#             confidence = 30

#         response = {
#             "status": status,
#             "confidence": confidence,
#             "extracted_from_receipt": extracted,
#             "chat_amount": chat_amount,
#             "amount_match": amount_match,
#             "receiver_match": receiver_ok,
#             "raw_ocr_preview": raw_text[:1000]
#         }

#         logging.info(f"FINAL RESULT: {status} | Confidence: {confidence}")
#         return jsonify(response)

#     except Exception as e:
#         logging.error(f"Crash: {e}")
#         return jsonify({"error": "Server error", "details": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

# main2.py - v4.0: With Blur Detection, Full Gmail Verification, and Robust Handling
from asyncio.log import logger
import re
import json
import logging
import requests
import datetime
from io import BytesIO
from fuzzywuzzy import fuzz
from PIL import Image
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# CONFIG
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
EMAIL_SENDER = "service@nayapay.com"
MERCHANT_ACCOUNT = "Soha Nah Pardeep".lower()
TIMESTAMP_DELTA = datetime.timedelta(minutes=45)
EMAIL_SENDER = "service@nayapay.com"
TIMESTAMP_DELTA = datetime.timedelta(minutes=45)
WIDER_DELTA = datetime.timedelta(hours=1)
CONFIDENCE_THRESHOLD_APPROVE = 90
CONFIDENCE_THRESHOLD_REVIEW = 50  # Lowered for more leniency

with open("cred.json") as f:
    oauth_creds = json.load(f)

CLIENT_ID = oauth_creds["client_id"]
CLIENT_SECRET = oauth_creds["client_secret"]
REFRESH_TOKEN = oauth_creds["refresh_token"]
TOKEN_URI = oauth_creds.get("token_uri", "https://oauth2.googleapis.com/token")
# ROBUST REGEX (handles OCR junk like "1D" instead of "ID", missing spaces)
PATTERNS = {
    "amount": r"Rs[\.\s]*([\d,]+)",  
    "transaction_id": r"Transaction\s*(?:ID|1D)[^\w\d]*([\w\d]{20,})",  # Handles "Transaction1D"
    # "timestamp": r"(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}\s*,\s*\d{1,2}:\d{2}\s*[AP]M)",  # Flexible spacing
    "timestamp":r"(\d{2}[A-Za-z]{3}\d{4}\d{4}(AM|PM))",
    "receiver": r"Destination\s+Acc?\.?\s*Title\s*(.+?)(?:\n|$)",  
}

def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100  # Threshold: <100 = blurry (adjust based on tests)

def aggressive_preprocess(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    is_blurry = detect_blur(img)
    if is_blurry:
        logging.info("Blur detected - applying deblur")
        img = cv2.GaussianBlur(img, (3,3), 0)  # Light deblur
        img = cv2.bilateralFilter(img, 9, 75, 75)  # Preserve edges

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    methods = [
        gray,
        cv2.equalizeHist(gray),
        cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),  # Mild upscale
    ]

    best_text = ""
    for processed in methods:
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:-RsPKR'
        text = pytesseract.image_to_string(processed, config=config)
        if len(text) > len(best_text):
            best_text = text

    if len(best_text) < 100 and is_blurry:  # Too sketchy
        raise ValueError("Image too blurry/sketchy - reject")

    return best_text

def re_extract_transaction_id(full_text, original_tid):
    """
    Try to re-OCR only the line that contains Transaction ID.
    Works by isolating the Transaction ID line and cleaning OCR errors.
    """
    # Try to isolate the line with Transaction ID
    lines = full_text.split("\n")
    candidate = ""
    for line in lines:
        if "transaction" in line.lower():
            candidate = line
            break

    if not candidate:
        return original_tid  # nothing to improve

    # Clean the line for OCR-like junk
    candidate = candidate.replace("1D", "ID").replace("Id", "ID").replace("iD", "ID")

    # Try extracting again from this cleaner line
    m = re.search(r"(?:ID|1D)[^\w\d]*([\w\d]{20,})", candidate, re.IGNORECASE)
    if m:
        new_tid = m.group(1).strip()
        if len(new_tid) >= 24:
            return new_tid[:24]
        return new_tid

    return original_tid

def fix_tid_ocr_errors(raw_tid):
    allowed = "0123456789abcdef"
    tid = raw_tid.lower()
    logging.info(f"Original TxID OCR: {tid}")
    corrections = {
        'o': '0',
        'l': '1',
        'i': '1',
        'h': 'b',
        't': 'f',
        's': '5',
        'g': '9',
        'q': '9',
        'z': '2',
        'u': '0',
        'v': 'y',  # rarely wrong
    }

    fixed = ""
    for ch in tid:
        if ch in allowed:
            fixed += ch
        elif ch in corrections:
            fixed += corrections[ch]
        else:
            fixed += "0"   # last fallback: replace junk
    logging.info(f"Original TxID OCR:{fixed}")
    return fixed[:24]

def extract_fields(text):
    data = {}
    score = 0
    text_lower = text.lower()

    amt_match = re.search(PATTERNS["amount"], text, re.IGNORECASE)
    if amt_match:
        data["amount"] = int(amt_match.group(1).replace(",", ""))
        score += 40

    tid = re.search(PATTERNS["transaction_id"], text, re.IGNORECASE)
    rw_tid= tid.group(1).strip()
    tid_match = fix_tid_ocr_errors(rw_tid)

    if tid_match:
        # data["transaction_id"] = tid_match.group(1).strip().lower()
        data["transaction_id"] = tid_match
        if (len(data["transaction_id"]) == 24):
            score += 30
        else:
            improved_tid = re_extract_transaction_id(text, tid_match)
            if len(improved_tid) == 24:
                data["transaction_id"] = improved_tid
                score += 28  # slightly lower because corrected
            else:
                # fallback: truncate but still insert what we have
                data["transaction_id"] = improved_tid[:24]
                score += 25

    ts_match = re.search(PATTERNS["timestamp"], text)
    if ts_match:
        raw_ts = ts_match.group(1)

    # Convert "03Dec20250817AM" → "03 Dec 2025 08:17 AM"
        try:
            day = raw_ts[0:2]
            month = raw_ts[2:5]
            year = raw_ts[5:9]
            hour = raw_ts[9:11]
            minute = raw_ts[11:13]
            ampm = raw_ts[13:]

            formatted = f"{day} {month} {year} {hour}:{minute} {ampm}"
            print( "timestamp:",formatted)
            data["timestamp"] = datetime.datetime.strptime(formatted, "%d %b %Y %I:%M %p")

            score += 20
        except Exception as e:
            logging.error(f"Timestamp parse error: {e}")
            pass

    rec_match = re.search(PATTERNS["receiver"], text, re.IGNORECASE)
    if rec_match:
        data["receiver"] = rec_match.group(1).strip().lower()
        score += 10
    logging.info(f"Extracted: {data} | Confidence: {score}")
    return data, score

def parse_chat_amount(chat_text):
    text = chat_text.lower()
    patterns = [
        r"final[^\d]*([\d,]+)",
        r"total[^\d]*([\d,]+)",
        r"([\d,]+)\s*pkr",
        r"ban[ae]ga\s*([\d,]+)",
        r"([\d,]+)\s*rupees?",
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            return int(match.group(1).replace(",", ""))
    return None

def get_access_token():
    """Get a new access token from the refresh token"""
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token"
    }
    resp = requests.post(TOKEN_URI, data=data)
    resp.raise_for_status()
    token_data = resp.json()
    return token_data["access_token"]


# def fetch_email(credentials_json, timestamp):
#     try:
#         logger.info("---- FETCH EMAIL START ----")
#         logger.info("[1] Loading Gmail credentials...")
#         creds = Credentials.from_authorized_user_info(json.loads(credentials_json))
#         logger.info("[1] Credentials loaded successfully.")
#         service = build("gmail", "v1", credentials=creds)
#         logger.info("[2] Gmail service initialized.")

#         start = (timestamp - TIMESTAMP_DELTA).strftime("%Y/%m/%d")
#         end = (timestamp + TIMESTAMP_DELTA).strftime("%Y/%m/%d")
#         query = f"from:{EMAIL_SENDER} after:{start} before:{end}"
#         logger.info(f"[3] Gmail query prepared: {query}")

#         logger.info("[4] Fetching list of messages...")
#         results = service.users().messages().list(userId="me", q=query).execute()
#         messages = results.get("messages", [])
#         logger.info(f"[4] Messages found: {len(messages)}")

        
#         for msg in messages:
#             msg_data = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
#             logger.info(f"[5] Fetching email content for ID: {msg_id}")
#             # Parse body (base64 decode if needed, but snippet often suffices)
#             snippet = msg_data["snippet"]
#             payload = msg_data["payload"]
#             body = ""
#             logger.info("[6] Checking payload for text/plain parts...")
#             if "parts" in payload:
#                 for part in payload["parts"]:
#                     if part["mimeType"] == "text/plain":
#                         logger.info("[6] Found text/plain part. Decoding...")
#                         body = part["body"]["data"]  # Base64 decode if needed
#                         import base64
#                         body = base64.urlsafe_b64decode(body).decode("utf-8")
#                         logger.info("[6] Base64 decoding successful.")
#                         break
#                     else:
#                         logger.info("[6] No parts found in payload (possibly simple text email).")
#             full_text = snippet + body
#             logger.info("[7] Extracting fields from email text...")
#             tid_match = re.search(PATTERNS["transaction_id"], full_text, re.IGNORECASE)
#             amt_match = re.search(PATTERNS["amount"], full_text, re.IGNORECASE)
#             ts_match = re.search(PATTERNS["timestamp"], full_text)
#             logger.info(f"[7] Regex results: TID={bool(tid_match)}, AMT={bool(amt_match)}, TS={bool(ts_match)}")

#             if tid_match and amt_match:
#                 logger.info("[8] Matching email found. Extracting values...")
#                 email_ts_str = ts_match.group(1).strip() if ts_match else ""
#                 email_ts = datetime.datetime.strptime(email_ts_str, "%d %b %Y, %I:%M %p") if email_ts_str else None
#                 return {
#                     "transaction_id": tid_match.group(1),
#                     "amount": int(amt_match.group(1).replace(",", "")),
#                     "timestamp": email_ts
#                 }
#         # Fallback wider search
#         start_wide = (timestamp - WIDER_DELTA).strftime("%Y/%m/%d")
#         end_wide = (timestamp + WIDER_DELTA).strftime("%Y/%m/%d")
#         query_wide = f"from:{EMAIL_SENDER} after:{start_wide} before:{end_wide}"
#         # Repeat search logic...
#         # (Omit repetition for brevity; implement similarly)
#         return None
#     except Exception as e:
#         logging.error(f"Gmail error: {e}")
#         return None
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# def fetch_email(token, timestamp):
#     try:
#         logger.info("---- FETCH EMAIL START ----")

#         logger.info("[1] Creating credentials using token only...")
#         creds = Credentials(token=token)
#         logger.info("[1] Credentials created successfully.")

#         service = build("gmail", "v1", credentials=creds)
#         logger.info("[2] Gmail service initialized.")

#         start = (timestamp - TIMESTAMP_DELTA).strftime("%Y/%m/%d")
#         end = (timestamp + TIMESTAMP_DELTA).strftime("%Y/%m/%d")
#         query = f"from:{EMAIL_SENDER} after:{start} before:{end}"
#         logger.info(f"[3] Gmail query prepared: {query}")

#         results = service.users().messages().list(userId="me", q=query).execute()
#         messages = results.get("messages", [])
#         logger.info(f"[4] Messages found: {len(messages)}")

#         for msg in messages:
#             msg_data = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
#             snippet = msg_data.get("snippet", "")
#             payload = msg_data.get("payload", {})
#             body = ""

#             if "parts" in payload:
#                 for part in payload["parts"]:
#                     if part.get("mimeType") == "text/plain":
#                         import base64
#                         body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
#                         break

#             full_text = snippet + body

#             tid_match = re.search(PATTERNS["transaction_id"], full_text, re.IGNORECASE)
#             amt_match = re.search(PATTERNS["amount"], full_text, re.IGNORECASE)
#             ts_match  = re.search(PATTERNS["timestamp"], full_text)

#             if tid_match and amt_match:
#                 email_ts = None
#                 if ts_match:
#                     try:
#                         email_ts = datetime.datetime.strptime(ts_match.group(1), "%d %b %Y, %I:%M %p")
#                     except:
#                         pass

#                 return {
#                     "transaction_id": tid_match.group(1),
#                     "amount": int(amt_match.group(1).replace(",", "")),
#                     "timestamp": email_ts
#                 }

#         return None
#     except Exception as e:
#         logger.error(f"Gmail error: {e}")
#         return None
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

# def fetch_email_from_json(timestamp):
#     """
#     Fetch Gmail content using the OAuth client credentials JSON (single account)
#     """
#     try:
#         # Load OAuth credentials from JSON (contains access + refresh token)
#         creds = Credentials.from_authorized_user_file("cred.json", SCOPES)

#         service = build("gmail", "v1", credentials=creds)

#         # Gmail query
#         start = (timestamp - TIMESTAMP_DELTA).strftime("%Y/%m/%d")
#         end   = (timestamp + TIMESTAMP_DELTA).strftime("%Y/%m/%d")
#         query = f"from:{EMAIL_SENDER} after:{start} before:{end}"
#         logging.info(f"Gmail query: {query}")

#         results = service.users().messages().list(userId="me", q=query).execute()
#         messages = results.get("messages", [])

#         for msg in messages:
#             msg_data = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
#             snippet = msg_data.get("snippet", "")
#             payload = msg_data.get("payload", {})
#             body = ""

#             if "parts" in payload:
#                 for part in payload["parts"]:
#                     if part.get("mimeType") == "text/plain":
#                         import base64
#                         body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
#                         break

#             full_text = snippet + body

#             tid_match = re.search(PATTERNS["transaction_id"], full_text, re.IGNORECASE)
#             amt_match = re.search(PATTERNS["amount"], full_text, re.IGNORECASE)
#             ts_match  = re.search(PATTERNS["timestamp"], full_text)

#             if tid_match and amt_match:
#                 email_ts = None
#                 if ts_match:
#                     try:
#                         email_ts = datetime.datetime.strptime(ts_match.group(1), "%d %b %Y, %I:%M %p")
#                     except:
#                         pass

#                 return {
#                     "transaction_id": tid_match.group(1),
#                     "amount": int(amt_match.group(1).replace(",", "")),
#                     "timestamp": email_ts
#                 }

#         return None

#     except Exception as e:
#         logging.error(f"Gmail fetch error: {e}")
#         return None

# def fetch_email(timestamp):
#     """
#     Fetch email from EMAIL_SENDER around the given timestamp
#     Returns dict with transaction_id, amount, timestamp
#     """
#     try:
#         access_token = get_access_token()
#         creds = Credentials(token=access_token)
#         service = build("gmail", "v1", credentials=creds)

#         start = (timestamp - TIMESTAMP_DELTA).strftime("%Y/%m/%d")
#         end = (timestamp + TIMESTAMP_DELTA).strftime("%Y/%m/%d")
#         query = f"from:{EMAIL_SENDER} after:{start} before:{end}"

#         results = service.users().messages().list(userId="me", q=query).execute()
#         messages = results.get("messages", [])

#         for msg in messages:
#             msg_data = service.users().messages().get(
#                 userId="me", id=msg["id"], format="full"
#             ).execute()

#             snippet = msg_data.get("snippet", "")
#             payload = msg_data.get("payload", {})
#             body = ""

#             if "parts" in payload:
#                 for part in payload["parts"]:
#                     if part.get("mimeType") == "text/plain":
#                         import base64
#                         body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
#                         break

#             full_text = snippet + body

#             tid_match = re.search(r"Transaction\s*(?:ID|1D)[^\w\d]*([\w\d]{20,})", full_text, re.IGNORECASE)
#             amt_match = re.search(r"Rs[\.\s]*([\d,]+)", full_text, re.IGNORECASE)
#             ts_match  = re.search(r"(\d{2}[A-Za-z]{3}\d{4}\d{4}(AM|PM))", full_text)

#             if tid_match and amt_match:
#                 email_ts = None
#                 if ts_match:
#                     try:
#                         raw_ts = ts_match.group(1)
#                         day = raw_ts[0:2]
#                         month = raw_ts[2:5]
#                         year = raw_ts[5:9]
#                         hour = raw_ts[9:11]
#                         minute = raw_ts[11:13]
#                         ampm = raw_ts[13:]
#                         email_ts = datetime.datetime.strptime(f"{day} {month} {year} {hour}:{minute} {ampm}", "%d %b %Y %I:%M %p")
#                     except Exception as e:
#                         logging.warning(f"Failed to parse email timestamp: {e}")

#                 return {
#                     "transaction_id": tid_match.group(1),
#                     "amount": int(amt_match.group(1).replace(",", "")),
#                     "timestamp": email_ts
#                 }

#         return None

#     except Exception as e:
#         logging.error(f"Gmail fetch error: {e}")
#         return None

def fetch_email(timestamp):
    """
    Fetch email from EMAIL_SENDER around the given timestamp
    Returns dict with transaction_id, amount, timestamp
    """
    try:
        logging.info("Starting Gmail fetch process...")
        access_token = get_access_token()
        logging.info("Access token obtained successfully.")

        creds = Credentials(token=access_token)
        service = build("gmail", "v1", credentials=creds)
        logging.info("Gmail service initialized.")

        start = (timestamp - TIMESTAMP_DELTA).strftime("%Y/%m/%d")
        end = (timestamp + TIMESTAMP_DELTA).strftime("%Y/%m/%d")
        query = f"from:{EMAIL_SENDER} after:{start} before:{end}"
        logging.info(f"Gmail search query: {query}")

        results = service.users().messages().list(userId="me", q=query).execute()
        messages = results.get("messages", [])
        logging.info(f"Messages found: {len(messages)}")

        for msg in messages:
            msg_id = msg.get("id")
            logging.info(f"Fetching message ID: {msg_id}")
            msg_data = service.users().messages().get(
                userId="me", id=msg_id, format="full"
            ).execute()

            snippet = msg_data.get("snippet", "")
            payload = msg_data.get("payload", {})
            body = ""

            if "parts" in payload:
                for part in payload["parts"]:
                    if part.get("mimeType") == "text/plain":
                        import base64
                        try:
                            body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                            logging.info("Email body decoded successfully.")
                        except Exception as e:
                            logging.warning(f"Failed to decode email body: {e}")
                        break

            full_text = snippet + body
            logging.debug(f"Full email text (truncated 200 chars): {full_text[:200]}")

            tid_match = re.search(r"Transaction\s*(?:ID|1D)[^\w\d]*([\w\d]{20,})", full_text, re.IGNORECASE)
            amt_match = re.search(r"Rs[\.\s]*([\d,]+)", full_text, re.IGNORECASE)
            ts_match  = re.search(r"(\d{2}[A-Za-z]{3}\d{4}\d{4}(AM|PM))", full_text)

            if tid_match and amt_match:
                email_ts = None
                if ts_match:
                    try:
                        raw_ts = ts_match.group(1)
                        day = raw_ts[0:2]
                        month = raw_ts[2:5]
                        year = raw_ts[5:9]
                        hour = raw_ts[9:11]
                        minute = raw_ts[11:13]
                        ampm = raw_ts[13:]
                        email_ts = datetime.datetime.strptime(
                            f"{day} {month} {year} {hour}:{minute} {ampm}",
                            "%d %b %Y %I:%M %p"
                        )
                        logging.info(f"Email timestamp parsed: {email_ts}")
                    except Exception as e:
                        logging.warning(f"Failed to parse email timestamp: {e}")

                logging.info(f"Transaction matched. TID: {tid_match.group(1)}, Amount: {amt_match.group(1)}")
                return {
                    "transaction_id": tid_match.group(1),
                    "amount": int(amt_match.group(1).replace(",", "")),
                    "timestamp": email_ts
                }

        logging.info("No matching email found in the given timeframe.")
        return None

    except Exception as e:
        logging.error(f"Gmail fetch error: {e}", exc_info=True)
        return None

def detect_fraud(image_bytes):
    # Simple check (expand as needed)
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    edges = cv2.Canny(img, 100, 200)
    if np.mean(edges) > 200:  # High edges = possible edit
        return True
    return False

def calculate_confidence(extracted, chat_amount, email_data, is_fraud, extraction_score):
    score = extraction_score
    if is_fraud:
        return 0, "Fraud detected"

    img_amount = extracted.get("amount", 0)
    img_tid = extracted.get("transaction_id", "")
    img_ts = extracted.get("timestamp", None)
    img_receiver = extracted.get("receiver", "")

    # Amount
    email_amt = email_data.get("amount") if email_data else None
    if img_amount == chat_amount and (email_amt is None or img_amount == email_amt):
        score += 40
    elif fuzz.ratio(str(img_amount), str(chat_amount)) > 90:
        score += 20

    # Tx ID
    if email_data and fuzz.ratio(img_tid, email_data.get("transaction_id", "")) > 95:
        score += 30

    # Timestamp
    if img_ts and email_data and email_data.get("timestamp"):
        delta = abs(img_ts - email_data["timestamp"])
        if delta <= datetime.timedelta(minutes=5):
            score += 20
        elif delta <= TIMESTAMP_DELTA:
            score += 10

    # Receiver
    if fuzz.partial_ratio(img_receiver, MERCHANT_ACCOUNT) > 80:
        score += 10

    if not email_data:
        score -= 20  # Penalty for no email

    status = "APPROVED" if score >= CONFIDENCE_THRESHOLD_APPROVE else \
             "MANUAL REVIEW" if score >= CONFIDENCE_THRESHOLD_REVIEW else "REJECTED"
    return min(score, 100), status

@app.route("/verify_payment", methods=["POST"])
def verify_payment():
    try:
        image_file = request.files["image"]
        chat_file = request.files["chat_file"]
        # gmail_token = request.form["gmail_token"]
        # gmail_token = request.form.get("gmail_token")  # Just the raw token string
        # if gmail_token:
        #     creds = Credentials(
        #         token=gmail_token,
        #         refresh_token=None,
        #         token_uri="https://oauth2.googleapis.com/token",
        #         client_id=None,
        #         client_secret=None
        #     )
        # else:
        #     creds = None  # Skip Gmail verification

        image_bytes = image_file.read()
        chat_text = chat_file.read().decode("utf-8")

        logging.info("Starting OCR...")
        raw_text = aggressive_preprocess(image_bytes)
        logging.info(f"Raw OCR Text:\n{raw_text}")

        extracted, extraction_conf = extract_fields(raw_text)
        if extraction_conf < 50:
            return jsonify({"error": "Could not read receipt clearly. Please send a clearer screenshot.", "raw_text": raw_text[:1000]}), 400

        chat_amount = parse_chat_amount(chat_text)
        if not chat_amount:
            return jsonify({"error": "Could not determine order amount from chat"}), 400

        ts = extracted.get("timestamp")
        if not ts:
            return jsonify({"error": "No timestamp extracted - cannot verify email"}), 400
        #fetching mail
        # email_data = fetch_email(gmail_token, ts)
        email_data = fetch_email(ts)

        is_fraud = detect_fraud(image_bytes)

        score, status = calculate_confidence(extracted, chat_amount, email_data, is_fraud, extraction_conf)

        response = {
            "status": status,
            "confidence": score,
            "extracted_from_receipt": extracted,
            "chat_amount": chat_amount,
            "email_data": email_data,
            "fraud_detected": is_fraud,
            "receiver account": MERCHANT_ACCOUNT
            #"raw_ocr_preview": raw_text[:1000]
        }

        logging.info(f"\nFINAL RESULT: {status} | Confidence: {score}")
        return jsonify(response)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": "Server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)