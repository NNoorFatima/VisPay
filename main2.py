
from asyncio.log import logger
import base64
import re
from bs4 import BeautifulSoup
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
TXID_FUZZY_THRESHOLD = 80  # % similarity to consider as match
WIDER_DELTA = datetime.timedelta(hours=1)


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
    # logging.info(f"Original TxID OCR: {tid}")
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
    # logging.info(f"Original TxID OCR:{fixed}")
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
    """
    Parse the chat text and return the amount mentioned in it.

    Parameters:
    chat_text (str): The text to parse.

    Returns:
    int: The extracted amount, or None if no amount was found.

    The supported patterns are:
    - "final <amount>"
    - "total <amount>"
    - "<amount> kr"
    - "ban[ae]ga <amount>"
    - "<amount> rupees?"
    """
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
    """
    Reads credentials.json and returns a fresh access token using refresh_token
    """
    with open("cred.json", "r") as f:
        creds = json.load(f)

    data = {
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token"
    }

    token_resp = requests.post(creds["token_uri"], data=data)
    token_resp.raise_for_status()
    access_token = token_resp.json()["access_token"]
    logging.info("Access token obtained successfully.")
    return access_token

def fetch_email(transaction_id, timestamp):
    """
    Fetch email from EMAIL_SENDER on the same day as timestamp and matching TxID
    Returns dict with transaction_id, amount, timestamp
    """
    try:
        logging.info("---- Starting Gmail fetch process ----")
        access_token = get_access_token()
        creds = Credentials(token=access_token)
        service = build("gmail", "v1", credentials=creds)
        logging.info("Gmail service initialized.")

        day_start = timestamp.strftime("%Y/%m/%d")
        day_end = (timestamp + datetime.timedelta(days=1)).strftime("%Y/%m/%d")
        query = f"from:{EMAIL_SENDER} after:{day_start} before:{day_end}"
        logging.info(f"Gmail search query: {query}")

        results = service.users().messages().list(userId="me", q=query).execute()
        messages = results.get("messages", [])
        logging.info(f"Total messages found: {len(messages)}")

        def extract_parts(payload):
            """Recursively extract all text from payload parts"""
            text = ""
            email_timestamp = None
            if "parts" in payload:
                for idx, part in enumerate(payload["parts"]):
                    mime_type = part.get("mimeType", "")
                    logging.info(f"Checking part {idx + 1}: mimeType={mime_type}")
                    try:
                        if mime_type == "text/plain" and part.get("body", {}).get("data"):
                            decoded = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                            logging.info("Text/plain part decoded successfully.")
                            text += decoded + "\n"
                        elif mime_type == "text/html" and part.get("body", {}).get("data"):
                            html = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                            # --- Extract timestamp BEFORE cleaning ---
                            timestamp_regex_html = r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4},\s+\d{1,2}:\d{2}\s*(AM|PM)'
                            html_match = re.search(timestamp_regex_html, html)
                            if html_match:
                                raw_ts = html_match.group(0)
                                try:
                                    email_timestamp = datetime.datetime.strptime(raw_ts, "%d %b %Y, %I:%M %p")
                                except:
                                    email_timestamp = datetime.datetime.strptime(raw_ts, "%d %B %Y, %I:%M %p")

                            # --- Clean HTML for text extraction ---
                            soup = BeautifulSoup(html, "html.parser")
                            text += soup.get_text(separator=" ", strip=True) + "\n"
                            logging.info("Text/html part decoded and converted to plain text successfully.")
                        elif "parts" in part:
                            nested_text, nested_ts = extract_parts(part)
                            text += nested_text
                            if nested_ts:
                                email_timestamp = nested_ts
                    except Exception as e:
                        logging.warning(f"Failed to decode part {idx + 1} ({mime_type}): {e}")
            else:
                if payload.get("body", {}).get("data"):
                    decoded = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
                    text += decoded + "\n"
                    logging.info("Single-part payload decoded successfully.")
            return text, email_timestamp

        for i, msg in enumerate(messages):
            logging.info(f"Processing message {i + 1}/{len(messages)}: ID={msg['id']}")
            msg_data = service.users().messages().get(
                userId="me", id=msg["id"], format="full"
            ).execute()

            snippet = msg_data.get("snippet", "")
            payload = msg_data.get("payload", {})
            body_text, email_ts= extract_parts(payload)
            full_text = snippet + "\n" + body_text

            # logging.info(f"Full email text length: {len(full_text)}")
            # logging.info(f"\nFull email text content:\n{full_text}\n")

            # Extract TxID, Amount, Timestamp
            tid_match = re.search(r"Transaction\s*(?:ID|1D)[^\w\d]*([\w\d]{20,})", full_text, re.IGNORECASE)
            amt_match = re.search(r"Rs[\.\s]*([\d,]+)", full_text, re.IGNORECASE)
            # ts_match  = re.search(r"(\d{1,2}\s+\w+\s+\d{4},\s+\d{1,2}:\d{2}\s*(AM|PM))", full_text)

            if tid_match:
                extracted_tid = tid_match.group(1)
                similarity = fuzz.ratio(transaction_id.lower(), extracted_tid.lower())
                logging.info(f"TxID found: {extracted_tid} | Similarity with OCR TxID: {similarity}%")
            else:
                logging.info("No TxID found in this email.")
                continue

            if amt_match:
                logging.info(f"Amount found in email: {amt_match.group(1)}")
            else:
                logging.info("No amount found in this email.")
                continue
            logging.info(f"Evaluating similarity: {similarity} vs threshold {TXID_FUZZY_THRESHOLD}")
            if similarity >= TXID_FUZZY_THRESHOLD:
                logging.info(f"Matching email found: TxID={extracted_tid}, Amount={amt_match.group(1)}, Timestamp={email_ts}")
                return {
                    "transaction_id": extracted_tid,
                    "amount": int(amt_match.group(1).replace(",", "")),
                    "timestamp": email_ts  # Comes from raw HTML
                }

        logging.info("No matching email found in the given timeframe.")
        return None
    except Exception as e:
        logging.error(f"Gmail fetch error: {e}")
        return None

def detect_fraud(image_bytes):
    # Simple check (expand as needed)
    """
    Detects whether an image has been edited or tampered with.

    Currently, this function uses the Canny edge detection algorithm to detect
    high edges in the image, which may be indicative of image editing or
    tampering.

    :param image_bytes: The image to be analyzed
    :return: True if the image has been edited or tampered with, False otherwise
    """
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    edges = cv2.Canny(img, 100, 200)
    if np.mean(edges) > 200:  # High edges = possible edit
        return True
    return False

def calculate_confidence(extracted, chat_amount, email_data, is_fraud, extraction_score):
    """
    Calculate confidence score for a given extracted data from receipt image.

    Confidence score is calculated based on the following factors:

    1. Amount: If the amount from the image matches the chat amount, add 40. If the amount is similar (fuzz ratio > 90), add 20.
    2. Tx ID: If the Tx ID from the image matches the Tx ID from the email, add 30.
    3. Timestamp: If the timestamp from the image matches the timestamp from the email (within 5 minutes), add 20. If the timestamp is within the TIMESTAMP_DELTA, add 10.
    4. Receiver: If the receiver name from the image matches the merchant account name, add 10.
    If the email data is not provided, subtract 20 from the score.
    The final confidence score is capped at 100.
    Returns a tuple of (confidence_score, status) where status can be one of the following:

    APPROved: Confidence score >= CONFIDENCE_THRESHOLD_APPROVE
    manual_review: Confidence score >= CONFIDENCE_THRESHOLD_REVIEW
    rejected: Confidence score < CONFIDENCE_THRESHOLD_REVIEW

    :param extracted: Extracted data from receipt image
    :param chat_amount: Amount from chat
    :param email_data: Email data
    :param is_fraud: Whether the email is suspected to be fraudulent
    :param extraction_score: Extraction score from OCR
    :return: Tuple of (confidence_score, status)
    """
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
        tid= extracted.get("transaction_id")
        if not ts:
            return jsonify({"error": "No timestamp extracted - cannot verify email"}), 400
        #fetching mail
        # email_data = fetch_email(gmail_token, ts)
        email_data = fetch_email(tid, ts)

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