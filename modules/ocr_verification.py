"""
OCR verification module for payment receipt processing.
"""
import pytesseract
import re
from typing import Dict, Optional, Any
import cv2
import numpy as np
from rapidfuzz import fuzz
import uuid
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from modules.preprocessing import (
    preprocess_image_for_ocr_minimal,
    preprocess_image_for_ocr_light,
    preprocess_image_for_ocr,
    preprocess_image_for_ocr_medium,
    preprocess_image_for_ocr_advanced,
    preprocess_image_for_ocr_morphology,
    preprocess_image_for_ocr_scale_aware,
    preprocess_image_for_digital
)
from modules.image_analysis import auto_select_preprocessing
from config import settings

try:
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
except Exception:
    pass
# def extract_text_pytesseract(img: np.ndarray) -> str:
#     """
#     Extract text from image using pytesseract.
#     """
#     return pytesseract.image_to_string(img)

#receive text from config file 
# Replace or overload the existing helper
def extract_text_pytesseract(img: np.ndarray, config: str = "--oem 1 --psm 6") -> str:
    """
    Extract text from image using pytesseract with configurable options.
    Default: LSTM engine, assume a block of text.
    """
    # Ensure tesseract executable path is set (config.py should provide it)
    try:
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    except Exception:
        pass

    return pytesseract.image_to_string(img, config=config)

# def extract_text_easyocr(img: np.ndarray, reader) -> list:
#     """
#     Extract text from image using EasyOCR.
#     """
#     results = reader.readtext(img)
#     return results
#updated function 
def extract_text_easyocr(img: np.ndarray, reader, paragraph: bool = True, contrast_ths: float = 0.05, detail: int = 1) -> list:
    """
    Extract text from image using EasyOCR with configurable parameters.
    """
    results = reader.readtext(img, detail=detail, paragraph=paragraph, contrast_ths=contrast_ths)
    return results

def is_date_pattern(candidate: str) -> bool:
    """
    Check if a candidate string is actually a date pattern.
    """
    # Common date patterns
    date_patterns = [
        r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$',  # DD/MM/YYYY, MM/DD/YYYY
        r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{1,2}$',   # DD/MM/YY
        r'^\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}$',     # YYYY/MM/DD
        r'^\d{2}[/\-\.]\d{2}[/\-\.]\d{4}$',         # DD/MM/YYYY (strict)
        r'^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # DD Mon
        r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}',  # Mon DD
    ]
    
    for pattern in date_patterns:
        if re.match(pattern, candidate, re.IGNORECASE):
            # Additional validation: check if numbers are in valid date ranges
            parts = re.split(r'[/\-\.\s]', candidate)
            if len(parts) >= 2:
                try:
                    first = int(parts[0])
                    second = int(parts[1])
                    # If first part is 1-31 and second is 1-12, likely a date
                    if 1 <= first <= 31 and 1 <= second <= 12:
                        return True
                    # If first is 1-12 and second is 1-31, also likely a date
                    if 1 <= first <= 12 and 1 <= second <= 31:
                        return True
                except ValueError:
                    pass
            return True
    
    return False


def is_valid_transaction_id(candidate: str, context: str = "") -> bool:
    logger.debug(f"[VALIDATION] Checking candidate: '{candidate}' (context: '{context[:50]}...')")
    
    if not candidate or len(candidate) < 3:
        logger.debug(f"[VALIDATION] Rejected: too short or empty")
        return False
    
    label_words = {
        'RECEIPT', 'INVOICE', 'BILL', 'DOCUMENT', 'DOC', 'REFERENCE', 'REF',
        'ORDER', 'TRANSACTION', 'TRANS', 'TXN', 'PAYMENT', 'AMOUNT', 'TOTAL',
        'DATE', 'INVOICE NO', 'RECEIPT NO', 'BILL NO', 'DOC NO', 'REF NO',
        'ORDER NO', 'TRANSACTION ID', 'INVOICE NUMBER', 'BILL NUMBER',
        'RECEIPT NUMBER', 'DOCUMENT NUMBER', 'ORDER NUMBER', 'TRANS ID',
        'ID', 'NO', 'NUMBER', 'INV', 'NO.'
    }
    
    candidate_upper = candidate.upper().strip()
    
    if candidate_upper in label_words:
        logger.debug(f"[VALIDATION] Rejected: label word '{candidate_upper}'")
        return False
    
    for label in label_words:
        if candidate_upper.startswith(label + ' ') or candidate_upper == label:
            logger.debug(f"[VALIDATION] Rejected: starts with label word '{label}'")
            return False
    
    if not re.match(r'^[A-Z0-9\-/]+$', candidate_upper):
        logger.debug(f"[VALIDATION] Rejected: contains invalid characters")
        return False
    
    if is_date_pattern(candidate):
        logger.debug(f"[VALIDATION] Rejected: date pattern")
        return False
    
    if re.match(r'^\d+[.,]\d+$', candidate):
        logger.debug(f"[VALIDATION] Rejected: decimal number (amount)")
        return False
    
    is_pure_number = re.match(r'^\d+$', candidate)
    
    if is_pure_number:
        has_transaction_context = bool(
            context and re.search(
                r'(?:transaction|invoice|bill|receipt|doc|ref|order|id|no|number)',
                context.lower()
            )
        )
        if has_transaction_context:
            if len(candidate) >= 4:
                logger.debug(f"[VALIDATION] Accepted: pure number {len(candidate)} digits with transaction context")
                return True
        else:
            if len(candidate) >= 6:
                logger.debug(f"[VALIDATION] Accepted: pure number {len(candidate)} digits (no context)")
                return True
            else:
                logger.debug(f"[VALIDATION] Rejected: pure number {len(candidate)} digits too short without context")
                return False
    
    if re.match(r'^[A-Z]+$', candidate_upper):
        if candidate_upper in label_words:
            logger.debug(f"[VALIDATION] Rejected: label word '{candidate_upper}'")
            return False
        
        common_brands = {
            'TANTRA', 'SHOP', 'STORE', 'MALL', 'CENTER', 'CENTRE', 'MARKET',
            'SUPERMARKET', 'DEPARTMENT', 'OUTLET', 'BRANCH', 'LOCATION'
        }
        if candidate_upper in common_brands:
            logger.debug(f"[VALIDATION] Rejected: common brand/store name '{candidate_upper}'")
            return False
        
        if len(candidate_upper) < 6:
            logger.debug(f"[VALIDATION] Rejected: short all-letter word (likely label)")
            return False
        
        has_transaction_context = bool(
            context and re.search(
                r'(?:transaction|invoice|bill|receipt|doc|ref|order|id|no|number|document)',
                context.lower()
            )
        )
        
        if has_transaction_context and len(candidate_upper) >= 8:
            logger.debug(f"[VALIDATION] Accepted: long all-letter word {len(candidate_upper)} chars with transaction context")
            return True
        else:
            logger.debug(f"[VALIDATION] Rejected: all-letter word '{candidate_upper}' lacks transaction context (likely brand/store name)")
            return False
    
    if re.search(r'[A-Z]', candidate_upper) and re.search(r'[0-9]', candidate_upper):
        logger.debug(f"[VALIDATION] Accepted: alphanumeric mix")
        return True
    
    if len(candidate) >= 6:
        logger.debug(f"[VALIDATION] Accepted: long alphanumeric {len(candidate)} chars")
        return True
    
    if len(candidate) >= 4 and ('-' in candidate or '/' in candidate):
        if not is_date_pattern(candidate):
            logger.debug(f"[VALIDATION] Accepted: has separators and not date-like")
            return True
    
    logger.debug(f"[VALIDATION] Rejected: didn't match any acceptance criteria")
    return False



def extract_transaction_id_improved(text: str) -> Optional[str]:
    """
    Improved transaction ID extraction with multiple strategies.
    Excludes dates and amounts. Includes detailed logging.
    """
    logger.info(f"[EXTRACTION] Starting transaction ID extraction")
    logger.debug(f"[EXTRACTION] OCR text preview: {text[:200]}...")
    
    if not text:
        logger.warning("[EXTRACTION] Empty text provided")
        return None
    
    # Normalize text for better matching
    text = text.replace('\r', '\n')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    logger.debug(f"[EXTRACTION] Found {len(lines)} non-empty lines")
    
    transaction_patterns = [
        r'Document\s*(?:No\.?|Ho\'s|Hos|Ho|Number|Num)\s*[:#]?\s*([A-Z0-9\-/]{4,})',
        r'(?:Transaction\s*ID|Trans\s*ID|Invoice\s*No\.?|Invoice\s*Number|Bill\s*No\.?|Bill\s*Number|Receipt\s*No\.?|Receipt\s*Number|Doc\s*No\.?|Document\s*No\.?|Ref\s*No\.?|Reference\s*No\.?|Order\s*No\.?|Order\s*Number|Order\s*ID|ID)\s*[:#\s]*\s*([A-Z0-9\-/]+)',
        r'(?:Transaction|Invoice|Bill|Receipt|Document|Reference|Order)\s*(?:ID|No\.?|Number)\s*([A-Z0-9\-/]+)',
        r'(?:Transaction\s*ID|Trans\s*ID|Invoice\s*No\.?|Bill\s*No\.?|Receipt\s*No\.?|Doc\s*No\.?)\s*[:#\s]*\s*\n\s*([A-Z0-9\-/]+)',
        r'(?:INV|INVOICE|BILL|RECEIPT|DOC|REF|ORDER|TRANS|TXN)\s*[:#]?\s*([A-Z0-9\-/]{3,})',
        r'Document\s*[:#]?\s*([A-Z0-9\-/]{4,})',
    ]
    
    logger.info("[EXTRACTION] Strategy 1: Trying regex patterns...")
    # Try regex patterns first
    for idx, pattern in enumerate(transaction_patterns, 1):
        matches = re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        for match in matches:
            candidate = match.group(1).strip()
            logger.debug(f"[EXTRACTION] Pattern {idx} found candidate: '{candidate}'")
            match_start = match.start()
            match_end = match.end()
            context = text[max(0, match_start - 50):min(len(text), match_end + 50)]
            if is_valid_transaction_id(candidate, context=context):
                logger.info(f"[EXTRACTION] Accepted transaction ID: '{candidate}' (from pattern {idx})")
                candidate = candidate.replace('O', '0').replace('I', '1').replace('S', '5')
                return candidate.upper()
            else:
                logger.debug(f"[EXTRACTION] Pattern {idx} candidate rejected: '{candidate}'")
    
    logger.info("[EXTRACTION] Strategy 2: Searching lines with transaction keywords...")
    transaction_keywords = [
        'transaction id', 'trans id', 'invoice no', 'invoice number', 'bill no',
        'bill number', 'receipt no', 'receipt number', 'doc no', 'document no',
        'document', 'ref no', 'reference', 'order no', 'order number', 'order id'
    ]
    
    document_patterns = [
        r'document\s*(?:no|ho\'s|hos|ho|number|num)\s*[:#]?\s*([A-Z0-9\-/]{3,})',
        r'document\s*[:#]?\s*([A-Z0-9\-/]{3,})',  
    ]
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        logger.debug(f"[EXTRACTION] Checking line {i+1}: '{line[:100]}'")
        
        if 'document' in line_lower:
            logger.debug(f"[EXTRACTION] Found 'document' in line {i+1}, trying OCR-error-tolerant patterns...")
            for pattern in document_patterns:
                match = re.search(pattern, line_lower)
                if match:
                    candidate = match.group(1).strip().upper()
                    logger.debug(f"[EXTRACTION] Found candidate via document pattern: '{candidate}'")
                    if is_valid_transaction_id(candidate, context=line):
                        logger.info(f"[EXTRACTION] Accepted transaction ID: '{candidate}' (from document pattern, line {i+1})")
                        return candidate
        
        for keyword in transaction_keywords:
            if keyword in line_lower:
                logger.debug(f"[EXTRACTION] Found keyword '{keyword}' in line {i+1}")
                idx = line_lower.find(keyword)
                after_keyword = line[idx + len(keyword):].strip()
                after_keyword = re.sub(r'^[:#\s\-]+', '', after_keyword)
                logger.debug(f"[EXTRACTION] Text after keyword: '{after_keyword[:50]}'")
                
                id_matches = list(re.finditer(r'([A-Z0-9\-/]{4,})', after_keyword, re.IGNORECASE))
                if id_matches:
                    id_matches.sort(key=lambda m: len(m.group(1)), reverse=True)
                    id_match = id_matches[0]
                else:
                    id_match = re.search(r'([A-Z0-9\-/]{3,})', after_keyword, re.IGNORECASE)
                
                if not id_match:
                    id_match = re.search(r'(\d{4,})', after_keyword)  # 4+ digits
                
                if id_match:
                    candidate = id_match.group(1).strip().upper()
                    logger.debug(f"[EXTRACTION] Extracted candidate from same line: '{candidate}'")
                    # Use line as context
                    if is_valid_transaction_id(candidate, context=line):
                        logger.info(f"[EXTRACTION] Accepted transaction ID: '{candidate}' (from line {i+1})")
                        return candidate
                
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    logger.debug(f"[EXTRACTION] Checking next line: '{next_line[:50]}'")
                    id_match = re.search(r'^([A-Z0-9\-/]{3,})', next_line, re.IGNORECASE)
                    if not id_match:
                        id_match = re.search(r'^(\d{4,})', next_line)
                    
                    if id_match:
                        candidate = id_match.group(1).strip().upper()
                        logger.debug(f"[EXTRACTION] Extracted candidate from next line: '{candidate}'")
                        context = f"{line} {next_line}"
                        if is_valid_transaction_id(candidate, context=context):
                            logger.info(f"[EXTRACTION] Accepted transaction ID: '{candidate}' (from line {i+2})")
                            return candidate
    
    logger.info("[EXTRACTION] Strategy 3: Looking for standalone ID patterns...")
    for i, line in enumerate(lines):
        line_clean = line.strip().upper()
        if re.match(r'^[A-Z0-9\-/]{4,}$', line_clean):
            has_numbers = bool(re.search(r'[0-9]', line_clean))
            
            logger.debug(f"[EXTRACTION] Found standalone pattern in line {i+1}: '{line_clean}' (has numbers: {has_numbers})")
            
            context_lines = []
            if i > 0:
                context_lines.append(lines[i-1])
            context_lines.append(line)
            if i + 1 < len(lines):
                context_lines.append(lines[i+1])
            context = ' '.join(context_lines)
            
            if not has_numbers and not re.search(
                r'(?:document|transaction|invoice|bill|receipt|doc|ref|order|id|no|number)',
                context.lower()
            ):
                logger.debug(f"[EXTRACTION] Skipping pure letter word '{line_clean}' without transaction context")
                continue
            
            if is_valid_transaction_id(line_clean, context=context):
                logger.info(f"[EXTRACTION] Accepted transaction ID: '{line_clean}' (standalone pattern, line {i+1})")
                return line_clean
    
    logger.info("[EXTRACTION] Strategy 4: Looking for numeric transaction IDs...")
    for i, line in enumerate(lines):
        line_clean = line.strip()
        numeric_match = re.match(r'^(\d{4,15})$', line_clean)
        if numeric_match:
            candidate = numeric_match.group(1)
            logger.debug(f"[EXTRACTION] Found numeric candidate in line {i+1}: '{candidate}'")
            
            context_lines = []
            if i > 0:
                context_lines.append(lines[i-1])
            context_lines.append(line)
            if i + 1 < len(lines):
                context_lines.append(lines[i+1])
            context = ' '.join(context_lines).lower()
            
            has_id_context = bool(
                re.search(
                    r'(?:transaction|invoice|bill|receipt|doc|ref|order|id|no|number)',
                    context
                )
            )
            
            has_amount_context = bool(
                re.search(
                    r'(?:total|amount|paid|sum|grand\s*total)',
                    context
                )
            )
            
            if has_id_context and not has_amount_context:
                logger.debug(f"[EXTRACTION] Numeric candidate '{candidate}' has ID context, validating...")
                if is_valid_transaction_id(candidate, context=context):
                    logger.info(f"[EXTRACTION] ✅ Accepted numeric transaction ID: '{candidate}' (line {i+1})")
                    return candidate
                else:
                    logger.debug(f"[EXTRACTION] Numeric candidate '{candidate}' rejected by validation")
            else:
                logger.debug(f"[EXTRACTION] Numeric candidate '{candidate}' lacks ID context or has amount context")
    
    logger.warning("[EXTRACTION] No valid transaction ID found after all strategies")
    logger.debug(f"[EXTRACTION] All lines checked: {lines}")
    logger.debug(f"[EXTRACTION] Full text for debugging:\n{text}")
    return None


def parse_receipt_pytesseract(text: str) -> Dict[str, Optional[str]]:
    """
    Parse receipt text to extract transaction_id, amount, and date.
    Improved with better transaction ID extraction and logging.
    """
    logger.info("[PARSING] Starting receipt parsing")
    logger.debug(f"[PARSING] Full OCR text:\n{text}\n{'='*60}")
    
    clean_text = text

    logger.info("[PARSING] Extracting date...")
    date = None
    date_patterns = [
        # Pattern 1: Label and date on same line
        r'(?:Date|Invoice\s*Date|Bill\s*Date|Issued\s*On|Payment\s*Date|Transaction\s*Date)\s*[:]?\s*([\d]{1,2}[/\-\.][\d]{1,2}[/\-\.][\d]{2,4})',
        # Pattern 2: Date formats (DD/MM/YYYY, MM/DD/YYYY, etc.)
        r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b',
        # Pattern 3: Date on line after label
        r'(?:Date|Invoice\s*Date)\s*[:]?\s*\n\s*([\d]{1,2}[/\-\.][\d]{1,2}[/\-\.][\d]{2,4})',
    ]
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, clean_text, flags=re.IGNORECASE | re.MULTILINE)
        for match in matches:
            date_candidate = match.group(1).strip()
            if date_candidate and is_date_pattern(date_candidate):
                date = date_candidate
                logger.info(f"[PARSING] Extracted date: '{date}'")
                break
        if date:
            break
    
    if not date:
        logger.warning("[PARSING] No date found")

    logger.info("[PARSING] Extracting transaction ID...")
    transaction_id = extract_transaction_id_improved(clean_text)
    
    if transaction_id and date and transaction_id == date:
        logger.warning(f"[PARSING] Rejected transaction ID '{transaction_id}' - matches extracted date")
        transaction_id = None  
    
    if transaction_id and is_date_pattern(transaction_id):
        logger.warning(f"[PARSING] Rejected transaction ID '{transaction_id}' - is a date pattern")
        transaction_id = None

    if transaction_id:
        logger.info(f"[PARSING] Final transaction ID: '{transaction_id}'")
    else:
        logger.warning("[PARSING] No transaction ID extracted")

    logger.info("[PARSING] Extracting amount...")
    amount_patterns = [
        r'(?:Amount|Total|Grand\s*Total|Amount\s*Paid|Net\s*Total|Subtotal|Sub\s*Total|Sum|Payment)\s*[:]?\s*\$?\s*([\d,]+\.?\d*)',
        r'[₹$€£]?\s*([\d,]+\.?\d{2})\s*(?:USD|EUR|GBP|INR)?',
        r'(?:Total|Amount)\s*[:]?\s*\n\s*\$?\s*([\d,]+\.?\d*)',
    ]

    def search_patterns(patterns, text):
        for p in patterns:
            matches = re.finditer(p, text, flags=re.IGNORECASE | re.MULTILINE)
            for match in matches:
                value = match.group(1).strip()
                if value:
                    return value
        return None

    amount = search_patterns(amount_patterns, clean_text)
    
    if amount:
        logger.info(f"[PARSING] Extracted amount: '{amount}'")
    else:
        logger.warning("[PARSING] No amount found")
    
    if not date:
        logger.debug("[PARSING] Re-searching for date...")
        date_patterns_search = [
            r'(?:Date|Invoice\s*Date|Bill\s*Date|Issued\s*On|Payment\s*Date|Transaction\s*Date)\s*[:]?\s*([\d]{1,2}[/\-\.][\d]{1,2}[/\-\.][\d]{2,4})',
            r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b',
        ]
        date = search_patterns(date_patterns_search, clean_text)
        if date:
            logger.info(f"[PARSING] Extracted date (second attempt): '{date}'")

    result = {
        "transaction_id": transaction_id,
        "amount": amount,
        "date": date
    }
    
    logger.info(f"[PARSING] Final parsed result: transaction_id={transaction_id}, amount={amount}, date={date}")
    
    return result


def extract_field_fuzzy(lines: list, keywords: list, threshold: int = 70, field_type: str = "transaction") -> Optional[str]:
    """
    Extract field using fuzzy keyword matching with improved logic.
    """
    if not lines:
        return None
    
    lines_clean = [line.strip() for line in lines if line.strip()]
    
    for i, line in enumerate(lines_clean):
        line_lower = line.lower()
        for kw in keywords:
            # Check if keyword appears in line (fuzzy or exact)
            kw_lower = kw.lower()
            if kw_lower in line_lower or fuzz.partial_ratio(line_lower, kw_lower) >= threshold:
                kw_pos = line_lower.find(kw_lower)
                if kw_pos >= 0:
                    after_keyword = line[kw_pos + len(kw_lower):].strip()
                    after_keyword = re.sub(r'^[:#\s\-]+', '', after_keyword)
                    
                    if field_type == "transaction":
                        id_match = re.search(r'([A-Z0-9\-/]{3,})', after_keyword, re.IGNORECASE)
                        if id_match:
                            candidate = id_match.group(1).strip().upper()
                            if is_valid_transaction_id(candidate):
                                return candidate
                    elif field_type == "amount":
                        amount_match = re.search(r'([\d,]+\.?\d*)', after_keyword)
                        if amount_match:
                            return amount_match.group(1).strip()
                    elif field_type == "date":
                        date_match = re.search(r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})', after_keyword)
                        if date_match:
                            return date_match.group(1).strip()
                
                if i + 1 < len(lines_clean):
                    next_line = lines_clean[i + 1].strip()
                    
                    if field_type == "transaction":
                        id_match = re.search(r'^([A-Z0-9\-/]{3,})', next_line, re.IGNORECASE)
                        if id_match:
                            candidate = id_match.group(1).strip().upper()
                            if is_valid_transaction_id(candidate):
                                return candidate
                    elif field_type == "amount":
                        amount_match = re.search(r'^([\d,]+\.?\d*)', next_line)
                        if amount_match:
                            return amount_match.group(1).strip()
                    elif field_type == "date":
                        date_match = re.search(r'^(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})', next_line)
                        if date_match:
                            return date_match.group(1).strip()
    
    return None


def parse_receipt_easyocr(ocr_results: list) -> Dict[str, Optional[str]]:
    """
    Parse receipt using EasyOCR results with improved fuzzy matching.
    """
    lines = [res[1] for res in ocr_results if res[1].strip()]
    
    full_text = '\n'.join(lines)

    transaction_keywords = [
        "document no", "document number", "invoice id", "invoice no", "invoice number",
        "bill id", "bill no", "bill number", "transaction id", "trans id", "receipt no",
        "receipt number", "receipt id", "doc no", "doc number", "ref no", "reference no",
        "order no", "order number", "order id", "ref", "reference", "ticket no"
    ]
    amount_keywords = [
        "amount", "total", "amt", "total amount", "grand total", "net total",
        "subtotal", "sub total", "sum", "payment", "paid", "balance"
    ]
    date_keywords = [
        "date", "invoice date", "bill date", "payment date", "transaction date",
        "issued on", "issued date", "due date"
    ]

    transaction_id_regex = extract_transaction_id_improved(full_text)
    
    transaction_id_fuzzy = extract_field_fuzzy(lines, transaction_keywords, threshold=70, field_type="transaction")
    amount = extract_field_fuzzy(lines, amount_keywords, threshold=70, field_type="amount")
    date = extract_field_fuzzy(lines, date_keywords, threshold=70, field_type="date")
    
    transaction_id = transaction_id_regex or transaction_id_fuzzy

    return {
        "transaction_id": transaction_id,
        "amount": amount,
        "date": date
    }


def save_processed_image(processed_img: np.ndarray, image_id: str) -> str:
    """
    Save preprocessed image to processed folder with unique ID.
    """
    processed_dir = Path(settings.PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_id}_{timestamp}.jpg"
    filepath = processed_dir / filename
    
    cv2.imwrite(str(filepath), processed_img)
    
    return str(filepath)


def get_preprocessing_function(method: str):
    """
    Get the appropriate preprocessing function based on method name.
    """
    method_map = {
        "minimal": preprocess_image_for_ocr_minimal,
        "light": preprocess_image_for_ocr_light,
        "medium": preprocess_image_for_ocr_medium,
        "advanced": preprocess_image_for_ocr_advanced,
        "morphology": preprocess_image_for_ocr_morphology,
        "digital": preprocess_image_for_digital,
        "scale_aware": preprocess_image_for_ocr_scale_aware,
        "auto": None,  
        "default": preprocess_image_for_ocr,
        "old": preprocess_image_for_ocr,
    }
    
    return method_map.get(method.lower(), preprocess_image_for_ocr_minimal)


def verify_payment_receipt(
    img: np.ndarray,
    use_easyocr: bool = False,
    easyocr_reader=None,
    save_processed: bool = True,
    preprocessing_method: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to verify payment receipt from image.
    """
    # Generate unique ID for this processing session
    image_id = str(uuid.uuid4())[:8]  # Short UUID (8 characters)
    processed_image_path = None
    
    try:
        quality_metrics = {}
        
        should_auto_select = (
            (preprocessing_method is None and settings.OCR_AUTO_PREPROCESSING) or
            (preprocessing_method and preprocessing_method.lower() == "auto")
        )
        auto_selected_flag = should_auto_select
        
        if should_auto_select:
            method, quality_metrics = auto_select_preprocessing(img)
        elif preprocessing_method is None:
            # Use configured method
            method = settings.OCR_PREPROCESSING_METHOD
            print(method,"method used from settings\n")
        else:
            # Use explicitly provided method
            method = preprocessing_method
        
        preprocess_func = get_preprocessing_function(method)
        
        # Preprocess image
        processed_img = preprocess_func(img)
        
        # Save processed image if requested
        if save_processed:
            try:
                processed_image_path = save_processed_image(processed_img, image_id)
            except Exception as save_error:
                # Log but don't fail the entire process if saving fails
                print(f"Warning: Failed to save processed image: {save_error}")
        
        # Extract all OCR text
        raw_text = None
        ocr_results = None
        
        if use_easyocr and easyocr_reader:
            # ocr_results = extract_text_easyocr(processed_img, easyocr_reader)
            ocr_results = extract_text_easyocr(processed_img, easyocr_reader, paragraph=True, contrast_ths=0.05)
            # Combine all text
            raw_text = '\n'.join([res[1] for res in ocr_results if res[1].strip()])
        else:
            # raw_text = extract_text_pytesseract(processed_img)
            tess_config = "--oem 1 --psm 6"
            raw_text = extract_text_pytesseract(processed_img, config=tess_config)
        
        logger.info(f"[OCR] Extracted {len(raw_text)} characters of text")
        logger.debug(f"[OCR] Full OCR text:\n{raw_text}\n{'='*60}")
        
        # Use LLM for extraction (primary method)
        llm_confidence = {}
        llm_explanations = {}
        parsed = {}
        result_fields = {}
        
        try:
            from modules.llm_validation import extract_with_llm
            parsed = extract_with_llm(raw_text)
            
            # Check if LLM extraction succeeded
            if parsed.get("error"):
                logger.warning(f"[LLM] Extraction error: {parsed.get('error')}, falling back to regex...")
                # Fallback to regex extraction
                if use_easyocr and easyocr_reader and ocr_results:
                    parsed = parse_receipt_easyocr(ocr_results)
                else:
                    parsed = parse_receipt_pytesseract(raw_text)
                result_fields = parsed
            else:
                # LLM extraction succeeded
                result_fields = {
                    "transaction_id": parsed.get("transaction_id"),
                    "amount": parsed.get("amount"),
                    "date": parsed.get("date")
                }
                # Store LLM metadata
                llm_confidence = parsed.get("confidence", {})
                llm_explanations = parsed.get("explanations", {})
                
        except Exception as e:
            logger.error(f"[LLM] Failed to extract with LLM: {e}")
            # Fallback to basic extraction if LLM fails
            import traceback
            logger.error(traceback.format_exc())
            if use_easyocr and easyocr_reader and ocr_results:
                parsed = parse_receipt_easyocr(ocr_results)
            else:
                parsed = parse_receipt_pytesseract(raw_text)
            result_fields = parsed
        
        # Determine verification status
        verification_status = "verified" if all([
            result_fields.get("transaction_id"),
            result_fields.get("amount"),
            result_fields.get("date")
        ]) else "partial" if any([
            result_fields.get("transaction_id"),
            result_fields.get("amount"),
            result_fields.get("date")
        ]) else "failed"
        
        result = {
            **result_fields,
            "verification_status": verification_status,
            "extraction_method": "llm" if (parsed.get("extraction_method") == "llm" and not parsed.get("error")) else ("easyocr" if use_easyocr else "pytesseract"),
            "processing_id": image_id,
            "preprocessing_method": method,
            "auto_selected": auto_selected_flag,
            "llm_confidence": llm_confidence,
            "llm_explanations": llm_explanations
        }
        
        # Include quality metrics if auto-selected
        if quality_metrics:
            result["image_quality"] = {
                "overall_score": round(quality_metrics.get("overall_quality", 0), 2),
                "noise_level": round(quality_metrics.get("noise_level", 0), 2),
                "contrast": round(quality_metrics.get("contrast", 0), 2),
                "brightness": round(quality_metrics.get("brightness", 0), 2),
                "blur_level": round(quality_metrics.get("blur_level", 0), 2),
                "resolution_score": round(quality_metrics.get("resolution_score", 0), 2)
            }
        
        # Add processed image path if saved
        if processed_image_path:
            result["processed_image_path"] = processed_image_path
            result["processed_image_url"] = f"/processed/{Path(processed_image_path).name}"
        
        return result
    
    except Exception as e:
        return {
            "transaction_id": None,
            "amount": None,
            "date": None,
            "verification_status": "error",
            "error": str(e),
            "processing_id": image_id
        }

