# """
# LLM-based extraction and categorization for receipt fields.
# Primary method: Extract all OCR text, feed to LLM for categorization.
# Uses Google Gemini API (free tier).
# """
# import json
# import re
# from typing import Dict, Optional, Any, List
# import logging
# from config import settings
# import numpy as np

# logger = logging.getLogger(__name__)

# try:
#     import google.generativeai as genai
#     GEMINI_AVAILABLE = True
# except ImportError:
#     GEMINI_AVAILABLE = False
#     logger.warning("Google Generative AI (Gemini) not available. Install with: pip install google-generativeai")


# def extract_with_llm_and_authenticity_check(
#     raw_text: str,
#     img: Optional[np.ndarray] = None,
#     image_path: Optional[str] = None,
#     ocr_results: Optional[List[Any]] = None
# ) -> Dict[str, Any]:
#     """
#     Extract receipt fields AND check for image manipulation/authenticity.
    
#     Integrates with receipt_authenticity module for comprehensive fraud detection.
    
#     Args:
#         raw_text: OCR text extracted from image
#         img: Image as numpy array (for pixel-level forensics)
#         image_path: Path to image file (for EXIF analysis)
#         ocr_results: OCR results with confidence scores (for font consistency)
    
#     Returns:
#         Dictionary with extraction results + authenticity indicators
#     """
#     # First, extract fields using LLM
#     extraction_result = extract_with_llm(raw_text)
    
#     # Then, check authenticity
#     if img is not None:
#         try:
#             from modules.receipt_authenticity import check_image_authenticity
            
#             authenticity_result = check_image_authenticity(
#                 img=img,
#                 image_path=image_path,
#                 ocr_results=ocr_results,
#                 raw_ocr_text=raw_text
#             )
            
#             # Add authenticity data to extraction result
#             extraction_result["authenticity_check"] = authenticity_result
#             extraction_result["is_suspicious"] = authenticity_result.get("is_suspicious", False)
#             extraction_result["authenticity_score"] = authenticity_result.get("authenticity_score", 0.5)
#             extraction_result["authenticity_recommendation"] = authenticity_result.get("recommendation", "UNKNOWN")
            
#             logger.info(f"[LLM+AUTH] Extraction + Authenticity check complete. Auth score: {authenticity_result.get('authenticity_score')}")
            
#             return extraction_result
        
#         except Exception as e:
#             logger.warning(f"[LLM+AUTH] Authenticity check failed, returning extraction only: {e}")
#             return extraction_result
    
#     return extraction_result


# def extract_with_llm(raw_text: str) -> Dict[str, Any]:
#     """
#     Extract receipt fields directly from OCR text using Google Gemini API (free tier).
#     This is the primary extraction method.
#     """
#     if not GEMINI_AVAILABLE:
#         logger.error("Google Generative AI (Gemini) not available. Install with: pip install google-generativeai")
#         return {
#             "transaction_id": None,
#             "amount": None,
#             "date": None,
#             "confidence": {},
#             "error": "Gemini not available"
#         }
    
#     if not settings.GEMINI_API_KEY:
#         logger.error("GEMINI_API_KEY not set. Please set it in .env file.")
#         return {
#             "transaction_id": None,
#             "amount": None,
#             "date": None,
#             "confidence": {},
#             "error": "GEMINI_API_KEY not configured"
#         }
    
#     try:
#         logger.info("[LLM] Extracting fields from OCR text using Gemini...")
#         logger.debug(f"[LLM] OCR text length: {len(raw_text)} characters")
        
#         genai.configure(api_key=settings.GEMINI_API_KEY)
        
#         model = genai.GenerativeModel(settings.LLM_MODEL or "gemini-2.5-pro")
        
#         prompt = f"""Extract receipt data from this OCR text:

# {raw_text[:2000]}

# Extract these 3 fields:
# 1. transaction_id: Document/Invoice/Receipt number (alphanumeric code like "TD01167104", NOT words like "RECEIPT" or "INVOICE")
# 2. amount: Total amount paid (largest monetary value)
# 3. date: Transaction date (DD/MM/YYYY format)

# You are an expert in image authenticity and receipt fraud detection. 
# Analyze this receipt image and identify any signs of manipulation.
# Check for:
# - Digital edits (altered text, mismatched fonts, copy–paste edges, blur/sharpness differences).
# - Physical tampering (scratches, overwriting, heat marks, reprinting).
# - AI-generated or synthetic patterns (uniform textures, unrealistic shadows, repeated noise).
# - Financial fraud indicators (altered totals, dates, vendor name, currency values, taxes, QR/barcode tampering).

# Return your result in this exact JSON format:

# {
#   "manipulated": true/false,
#   "reason": "short explanation",
#   "confidence": 0-100,
#   "areas_of_concern": ["brief list of suspicious regions or fields"]
# }

# {{"transaction_id":"value_or_null","amount":"value_or_null","date":"value_or_null","confidence":{{"transaction_id":0.0,"amount":0.0,"date":0.0}},"authenticity_check":{{"amounts_consistent":true/false,"date_format_consistent":true/false,"formatting_typical":true/false}}}}"""

#         logger.debug("[LLM] Calling Gemini API...")
        
#         from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
#         safety_settings = {
#             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#         }
        
#         response = model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.1,  # Even lower for consistency
#                 max_output_tokens=2048,  # INCREASED from 800 to avoid truncation
#             ),
#             safety_settings=safety_settings
#         )
        
#         if not response.candidates or len(response.candidates) == 0:
#             logger.error("[LLM] No candidates returned from Gemini API")
#             logger.warning("[LLM] Safety block detected - trying with sanitized text...")
            
#             sanitized_text = _sanitize_ocr_text(raw_text)
#             if sanitized_text != raw_text:
#                 return extract_with_llm_sanitized(sanitized_text)
            
#             return {
#                 "transaction_id": None,
#                 "amount": None,
#                 "date": None,
#                 "confidence": {},
#                 "error": "No response candidates from API - safety block"
#             }
        
#         candidate = response.candidates[0]
        
#         if candidate.finish_reason == 2: 
#             logger.warning("[LLM] Response truncated due to MAX_TOKENS - retrying with sanitized text...")
#             sanitized_text = _sanitize_ocr_text(raw_text)
#             return extract_with_llm_sanitized(sanitized_text)
            
#         if candidate.finish_reason == 3: 
#             logger.warning("[LLM] Response blocked by safety filters")
#             logger.debug(f"[LLM] Safety ratings: {candidate.safety_ratings}")
            
#             sanitized_text = _sanitize_ocr_text(raw_text)
#             if sanitized_text != raw_text:
#                 logger.info("[LLM] Retrying with sanitized OCR text...")
#                 return extract_with_llm_sanitized(sanitized_text)
            
#             return {
#                 "transaction_id": None,
#                 "amount": None,
#                 "date": None,
#                 "confidence": {},
#                 "error": "Response blocked by safety filters",
#                 "safety_issue": True
#             }
        
#         if candidate.finish_reason not in [1, 2]:  # Not STOP or MAX_TOKENS
#             logger.warning(f"[LLM] Unusual finish_reason: {candidate.finish_reason}")
#             return {
#                 "transaction_id": None,
#                 "amount": None,
#                 "date": None,
#                 "confidence": {},
#                 "error": f"Unexpected finish_reason: {candidate.finish_reason}"
#             }
        
#         if not candidate.content or not candidate.content.parts:
#             logger.error("[LLM] No content parts in response")
#             logger.debug(f"[LLM] Candidate: {candidate}")
            
#             # Try with sanitized text
#             sanitized_text = _sanitize_ocr_text(raw_text)
#             if sanitized_text != raw_text:
#                 logger.info("[LLM] Retrying with sanitized OCR text...")
#                 return extract_with_llm_sanitized(sanitized_text)
            
#             return {
#                 "transaction_id": None,
#                 "amount": None,
#                 "date": None,
#                 "confidence": {},
#                 "error": "No content in response - likely safety block"
#             }
        
#         response_text = response.text.strip()
#         logger.debug(f"[LLM] Raw response: {response_text[:200]}...")
        
#         if "```json" in response_text:
#             response_text = response_text.split("```json")[1].split("```")[0].strip()
#         elif "```" in response_text:
#             response_text = response_text.split("```")[1].split("```")[0].strip()
        
#         json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#         if json_match:
#             response_text = json_match.group(0)
        
#         result = json.loads(response_text)
        
#         logger.info(f"[LLM] Extracted: transaction_id={result.get('transaction_id')}, amount={result.get('amount')}, date={result.get('date')}")
        
#         return {
#             "transaction_id": result.get("transaction_id"),
#             "amount": result.get("amount"),
#             "date": result.get("date"),
#             "confidence": result.get("confidence", {}),
#             "explanations": result.get("explanations", {}),
#             "corrected": True,
#             "extraction_method": "llm"
#         }
    
#     except json.JSONDecodeError as e:
#         logger.error(f"[LLM] Error parsing JSON response: {e}")
#         logger.error(f"[LLM] Response was: {response_text[:500]}")
#         return {
#             "transaction_id": None,
#             "amount": None,
#             "date": None,
#             "confidence": {},
#             "error": f"JSON parsing error: {str(e)}"
#         }
#     except Exception as e:
#         logger.error(f"[LLM] Error in LLM extraction (Gemini): {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return {
#             "transaction_id": None,
#             "amount": None,
#             "date": None,
#             "confidence": {},
#             "error": str(e)
#         }


# def categorize_field_simple(value: Optional[str], field_type: str) -> Dict[str, Any]:
#     """
#     Simple rule-based categorization (fallback when LLM not available)
#     """
#     if not value:
#         return {
#             "is_valid": False,
#             "category": "missing",
#             "confidence": 0.0
#         }
    
#     if field_type == "transaction_id":
#         # Check if it's a label word
#         label_words = {'RECEIPT', 'INVOICE', 'BILL', 'DOCUMENT', 'DOC', 'REFERENCE', 
#                       'REF', 'ORDER', 'TRANSACTION', 'TRANS', 'TXN', 'PAYMENT', 'ID'}
#         if value.upper().strip() in label_words:
#             return {
#                 "is_valid": False,
#                 "category": "label_word",
#                 "confidence": 0.0
#             }
        
#         # Check if it's a date pattern
#         from modules.ocr_verification import is_date_pattern
#         if is_date_pattern(value):
#             return {
#                 "is_valid": False,
#                 "category": "date_pattern",
#                 "confidence": 0.0
#             }
        
#         # Valid if it has alphanumeric mix or is long
#         has_letters = bool(re.search(r'[A-Z]', value.upper()))
#         has_numbers = bool(re.search(r'[0-9]', value))
#         is_long = len(value) >= 8
        
#         if (has_letters and has_numbers) or is_long:
#             return {
#                 "is_valid": True,
#                 "category": "transaction_id",
#                 "confidence": 0.8 if (has_letters and has_numbers) else 0.6
#             }
        
#         return {
#             "is_valid": False,
#             "category": "unknown",
#             "confidence": 0.3
#         }
    
#     return {
#         "is_valid": True,
#         "category": field_type,
#         "confidence": 0.7
#     }


# def _sanitize_ocr_text(text: str) -> str:
#     """
#     Sanitize OCR text to avoid triggering safety filters.
#     Removes potentially problematic characters while preserving structure.
#     """
#     # Remove non-ASCII characters that might trigger filters
#     sanitized = ''.join(char if ord(char) < 128 else ' ' for char in text)
    
#     # Remove excessive special characters
#     sanitized = re.sub(r'[^\w\s\.\,\:\-\/\(\)\$\€\£\@\#\%]', ' ', sanitized)
    
#     # Collapse multiple spaces
#     sanitized = re.sub(r'\s+', ' ', sanitized)
    
#     return sanitized.strip()


# def extract_with_llm_sanitized(sanitized_text: str) -> Dict[str, Any]:
#     """
#     Simplified version that uses already-sanitized text.
#     This is called when the first attempt fails due to safety filters or MAX_TOKENS.
#     """
#     if not GEMINI_AVAILABLE or not settings.GEMINI_API_KEY:
#         return {
#             "transaction_id": None,
#             "amount": None,
#             "date": None,
#             "confidence": {},
#             "error": "Cannot retry - Gemini not available"
#         }
    
#     try:
#         genai.configure(api_key=settings.GEMINI_API_KEY)
#         model = genai.GenerativeModel(settings.LLM_MODEL or "gemini-2.5-pro")
        
#         prompt = f"""Extract from OCR:

# {sanitized_text[:1500]}

# Find: transaction_id (like "TD01167104"), amount (total paid), date (DD/MM/YYYY)

# JSON only (no explanation):
# {{"transaction_id":"X","amount":"Y","date":"Z","confidence":{{"transaction_id":0.0,"amount":0.0,"date":0.0}}}}"""

#         from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
#         safety_settings = {
#             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#         }
        
#         response = model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.1, 
#                 max_output_tokens=2048  # INCREASED from 800
#             ),
#             safety_settings=safety_settings
#         )
        
#         if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
#             logger.error("[LLM] Sanitized attempt also failed - no content")
#             return {
#                 "transaction_id": None,
#                 "amount": None,
#                 "date": None,
#                 "confidence": {},
#                 "error": "Even sanitized text was blocked"
#             }
        
#         candidate = response.candidates[0]
        
#         if candidate.finish_reason == 2:
#             logger.error("[LLM] MAX_TOKENS hit even with sanitized text and 2048 tokens")
           
#             try:
#                 partial_text = response.text.strip()
#                 logger.debug(f"[LLM] Partial response: {partial_text}")
              
#                 if '{' in partial_text:
#                     partial_text = partial_text[partial_text.index('{'):]
                   
#                     if partial_text.count('{') > partial_text.count('}'):
#                         partial_text += '}'
#                     result = json.loads(partial_text)
#                     logger.warning("[LLM] Recovered partial JSON from truncated response")
#                     return {
#                         "transaction_id": result.get("transaction_id"),
#                         "amount": result.get("amount"),
#                         "date": result.get("date"),
#                         "confidence": result.get("confidence", {}),
#                         "corrected": True,
#                         "extraction_method": "llm_sanitized_partial"
#                     }
#             except:
#                 pass
            
#             return {
#                 "transaction_id": None,
#                 "amount": None,
#                 "date": None,
#                 "confidence": {},
#                 "error": "Response truncated even after optimization"
#             }
        
#         response_text = response.text.strip()
#         logger.debug(f"[LLM] Sanitized response: {response_text[:200]}...")
        
#         if "```json" in response_text:
#             response_text = response_text.split("```json")[1].split("```")[0].strip()
#         elif "```" in response_text:
#             response_text = response_text.split("```")[1].split("```")[0].strip()
        
#         json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#         if json_match:
#             response_text = json_match.group(0)
        
#         result = json.loads(response_text)
        
#         logger.info(f"[LLM] ✅ Extracted (sanitized): transaction_id={result.get('transaction_id')}, amount={result.get('amount')}, date={result.get('date')}")
        
#         return {
#             "transaction_id": result.get("transaction_id"),
#             "amount": result.get("amount"),
#             "date": result.get("date"),
#             "confidence": result.get("confidence", {}),
#             "explanations": result.get("explanations", {}),
#             "corrected": True,
#             "extraction_method": "llm_sanitized"
#         }
        
#     except json.JSONDecodeError as e:
#         logger.error(f"[LLM] JSON parsing error in sanitized extraction: {e}")
#         try:
#             logger.debug(f"[LLM] Failed to parse: {response_text[:500]}")
#         except:
#             pass
#         return {
#             "transaction_id": None,
#             "amount": None,
#             "date": None,
#             "confidence": {},
#             "error": f"JSON parsing failed: {str(e)}"
#         }
#     except Exception as e:
#         logger.error(f"[LLM] Sanitized extraction also failed: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return {
#             "transaction_id": None,
#             "amount": None,
#             "date": None,
#             "confidence": {},
#             "error": f"Sanitized extraction failed: {str(e)}"
#         }
"""
LLM-based extraction and categorization for receipt fields.
Primary method: Extract all OCR text, feed to LLM for categorization.
Uses Google Gemini API (free tier).
"""
import json
import re
from typing import Dict, Optional, Any, List
import logging
from config import settings
import numpy as np

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI (Gemini) not available. Install with: pip install google-generativeai")

# --- START: NEW JSON SCHEMA DEFINITION ---
JSON_SCHEMA_TEMPLATE = """{
    "transaction_id": "value_or_null",
    "amount": "value_or_null",
    "date": "value_or_null",
    "confidence": {
    "transaction_id": 0.0,
    "amount": 0.0,
    "date": 0.0
    },
    "authenticity_check": {
    "manipulated": false,
    "reason": "short explanation text (why you think manipulated or not)",
    "confidence": 0.0,
    "areas_of_concern": ["brief list of suspicious lines/fields or regions"],
    "notes": "optional short notes (max 200 chars)"
    }
}"""
# --- END: NEW JSON SCHEMA DEFINITION ---
def extract_with_llm_and_authenticity_check(
    raw_text: str,
    img: Optional[np.ndarray] = None,
    image_path: Optional[str] = None,
    ocr_results: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Extract receipt fields AND check for image manipulation/authenticity.
    
    Integrates with receipt_authenticity module for comprehensive fraud detection.
    
    Args:
        raw_text: OCR text extracted from image
        img: Image as numpy array (for pixel-level forensics)
        image_path: Path to image file (for EXIF analysis)
        ocr_results: OCR results with confidence scores (for font consistency)
    
    Returns:
        Dictionary with extraction results + authenticity indicators
    """
    # First, extract fields using LLM (LLM will also provide a text-based authenticity_check)
    extraction_result = extract_with_llm(raw_text)
    
    # Then, check authenticity with the dedicated image-forensics module if image data provided
    if img is not None:
        try:
            from modules.receipt_authenticity import check_image_authenticity
            
            authenticity_result = check_image_authenticity(
                img=img,
                image_path=image_path,
                ocr_results=ocr_results,
                raw_ocr_text=raw_text
            )
            
            # Merge LLM text-based authenticity_check (if present) with image-based results
            llm_auth = extraction_result.get("authenticity_check", {}) or {}
            merged_auth = dict(authenticity_result) if isinstance(authenticity_result, dict) else {}
            # keep the LLM text check under a separate key for traceability
            merged_auth["llm_text_check"] = llm_auth
            
            # Normalize authenticity confidence from LLM text check (0.0 - 1.0)
            llm_conf = llm_auth.get("confidence", 0.0)
            try:
                llm_conf = float(llm_conf)
                if llm_conf > 1.0:
                    llm_conf = llm_conf / 100.0
                llm_conf = max(0.0, min(1.0, llm_conf))
            except Exception:
                llm_conf = 0.0
            
            # Determine final authenticity score: prefer image-forensics score if present, else LLM text score
            image_score = None
            # try:
            image_score = merged_auth.get("authenticity_score", merged_auth.get("score", None))
            if image_score is not None:
                image_score = float(image_score)
                if image_score > 1.0:
                    image_score = image_score / 100.0
                image_score = max(0.0, min(1.0, image_score))    
            else:
                image_score = llm_conf # Fallback in case of module error
            # *** MODIFIED: Weighted Fusion (Prioritize Image Forensics 70% vs. LLM Text 30%) ***
            final_score = (image_score * 0.70) + (llm_conf * 0.30)
                 # --- END: MODIFIED SCORE FUSION LOGIC ---
            # except Exception:
            #     image_score = None
            
            # if image_score is not None:
            #     final_score = max(image_score, llm_conf)
            # else:
            #     final_score = llm_conf
            
            # Determine suspicious flag (respect image module's is_suspicious if it exists)
            is_suspicious_flag = bool(
                merged_auth.get("is_suspicious", merged_auth.get("manipulated", False)) 
                or llm_auth.get("manipulated", False)
            )
            
            # Attach merged authenticity info to extraction_result
            extraction_result["authenticity_check"] = merged_auth
            extraction_result["is_suspicious"] = extraction_result.get("is_suspicious", is_suspicious_flag)
            extraction_result["authenticity_score"] = extraction_result.get("authenticity_score", final_score)
            extraction_result["authenticity_recommendation"] = merged_auth.get("recommendation", merged_auth.get("notes", "UNKNOWN"))
            
            logger.info(f"[LLM+AUTH] Extraction + Authenticity check complete. Final auth score: {extraction_result.get('authenticity_score')}")
            
            return extraction_result
        
        except Exception as e:
            logger.warning(f"[LLM+AUTH] Authenticity check failed, returning extraction only: {e}")
            return extraction_result
    
    return extraction_result


def extract_with_llm(raw_text: str) -> Dict[str, Any]:
    """
    Extract receipt fields directly from OCR text using Google Gemini API (free tier).
    This is the primary extraction method.
    """
    if not GEMINI_AVAILABLE:
        logger.error("Google Generative AI (Gemini) not available. Install with: pip install google-generativeai")
        return {
            "transaction_id": None,
            "amount": None,
            "date": None,
            "confidence": {},
            "error": "Gemini not available"
        }
    
    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set. Please set it in .env file.")
        return {
            "transaction_id": None,
            "amount": None,
            "date": None,
            "confidence": {},
            "error": "GEMINI_API_KEY not configured"
        }
    
    try:
        logger.info("[LLM] Extracting fields from OCR text using Gemini...")
        logger.debug(f"[LLM] OCR text length: {len(raw_text)} characters")
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        model = genai.GenerativeModel(settings.LLM_MODEL or "gemini-2.5-pro")
        
        # Merged prompt: request extraction fields AND a text-based authenticity_check JSON
        prompt = f"""Extract receipt data from this OCR text:

{raw_text[:2000]}

Extract these 3 fields:
1. transaction_id: Document/Invoice/Receipt number (alphanumeric code like "TD01167104", NOT words like "RECEIPT" or "INVOICE")
2. amount: Total amount paid (largest monetary value)
3. date: Transaction date (DD/MM/YYYY format)

Also perform a text-based authenticity analysis (you will only see OCR text and any short forensic summary if provided). You are an expert in receipt fraud detection. Identify any signs of manipulation or inconsistency you can infer from the text and structure.

Check for (text/structure-based checks):
- Mathematical consistency (do line items, subtotals, tax, and total add up).
- Date consistency and format anomalies.
- Currency and symbol inconsistencies.
- Repeated or copied text patterns, mismatched labels, or formatting inconsistencies that suggest tampering.
- Unusual or suspicious vendor names, totals, or formatting.
- If provided, incorporate the short image_forensics_summary (EXIF, noise/cloning scores, OCR confidence per region) in your reasoning.

Return ONLY valid JSON (no markdown, no extra text). Use this exact schema:
{JSON_SCHEMA_TEMPLATE}""" # <--- MODIFIED: Using variable instead of literal f-string JSON
# {{
#   "transaction_id": "value_or_null",
#   "amount": "value_or_null",
#   "date": "value_or_null",
#   "confidence": {{
#     "transaction_id": 0.0,
#     "amount": 0.0,
#     "date": 0.0
#   }},
#   "authenticity_check": {{
#     "manipulated": false,
#     "reason": "short explanation text (why you think manipulated or not)",
#     "confidence": 0.0,
#     "areas_of_concern": ["brief list of suspicious lines/fields or regions"],
#     "notes": "optional short notes (max 200 chars)"
#   }}
# }}"""
        logger.debug("[LLM] Calling Gemini API...")
        
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Even lower for consistency
                max_output_tokens=2048,  # INCREASED from 800 to avoid truncation
            ),
            safety_settings=safety_settings
        )
        
        if not response.candidates or len(response.candidates) == 0:
            logger.error("[LLM] No candidates returned from Gemini API")
            logger.warning("[LLM] Safety block detected - trying with sanitized text...")
            
            sanitized_text = _sanitize_ocr_text(raw_text)
            if sanitized_text != raw_text:
                return extract_with_llm_sanitized(sanitized_text)
            
            return {
                "transaction_id": None,
                "amount": None,
                "date": None,
                "confidence": {},
                "error": "No response candidates from API - safety block"
            }
        
        candidate = response.candidates[0]
        
        if candidate.finish_reason == 2: 
            logger.warning("[LLM] Response truncated due to MAX_TOKENS - retrying with sanitized text...")
            sanitized_text = _sanitize_ocr_text(raw_text)
            return extract_with_llm_sanitized(sanitized_text)
            
        if candidate.finish_reason == 3: 
            logger.warning("[LLM] Response blocked by safety filters")
            logger.debug(f"[LLM] Safety ratings: {candidate.safety_ratings}")
            
            sanitized_text = _sanitize_ocr_text(raw_text)
            if sanitized_text != raw_text:
                logger.info("[LLM] Retrying with sanitized OCR text...")
                return extract_with_llm_sanitized(sanitized_text)
            
            return {
                "transaction_id": None,
                "amount": None,
                "date": None,
                "confidence": {},
                "error": "Response blocked by safety filters",
                "safety_issue": True
            }
        
        if candidate.finish_reason not in [1, 2]:  # Not STOP or MAX_TOKENS
            logger.warning(f"[LLM] Unusual finish_reason: {candidate.finish_reason}")
            return {
                "transaction_id": None,
                "amount": None,
                "date": None,
                "confidence": {},
                "error": f"Unexpected finish_reason: {candidate.finish_reason}"
            }
        
        if not candidate.content or not candidate.content.parts:
            logger.error("[LLM] No content parts in response")
            logger.debug(f"[LLM] Candidate: {candidate}")
            
            # Try with sanitized text
            sanitized_text = _sanitize_ocr_text(raw_text)
            if sanitized_text != raw_text:
                logger.info("[LLM] Retrying with sanitized OCR text...")
                return extract_with_llm_sanitized(sanitized_text)
            
            return {
                "transaction_id": None,
                "amount": None,
                "date": None,
                "confidence": {},
                "error": "No content in response - likely safety block"
            }
        
        response_text = response.text.strip()
        logger.debug(f"[LLM] Raw response: {response_text[:200]}...")
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        result = json.loads(response_text)
        
        # Extract standard fields
        transaction_id = result.get("transaction_id")
        amount = result.get("amount")
        date_val = result.get("date")
        confidence = result.get("confidence", {})
        
        # Extract LLM-provided authenticity_check (text-based)
        authenticity = result.get("authenticity_check", {}) or {}
        # Normalize authenticity confidence to 0.0 - 1.0 float
        try:
            auth_conf = float(authenticity.get("confidence", 0.0))
            if auth_conf > 1.0:
                auth_conf = auth_conf / 100.0
            auth_conf = max(0.0, min(1.0, auth_conf))
        except Exception:
            auth_conf = 0.0
        authenticity["confidence"] = auth_conf
        
        logger.info(f"[LLM] Extracted: transaction_id={transaction_id}, amount={amount}, date={date_val}")
        
        return {
            "transaction_id": transaction_id,
            "amount": amount,
            "date": date_val,
            "confidence": confidence,
            "explanations": result.get("explanations", {}),
            "authenticity_check": authenticity,
            "is_suspicious": bool(authenticity.get("manipulated", False)),
            "authenticity_score": auth_conf,
            "corrected": True,
            "extraction_method": "llm"
        }
    
    except json.JSONDecodeError as e:
        logger.error(f"[LLM] Error parsing JSON response: {e}")
        logger.error(f"[LLM] Response was: {response_text[:500]}")
        return {
            "transaction_id": None,
            "amount": None,
            "date": None,
            "confidence": {},
            "error": f"JSON parsing error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"[LLM] Error in LLM extraction (Gemini): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "transaction_id": None,
            "amount": None,
            "date": None,
            "confidence": {},
            "error": str(e)
        }


def categorize_field_simple(value: Optional[str], field_type: str) -> Dict[str, Any]:
    """
    Simple rule-based categorization (fallback when LLM not available)
    """
    if not value:
        return {
            "is_valid": False,
            "category": "missing",
            "confidence": 0.0
        }
    
    if field_type == "transaction_id":
        # Check if it's a label word
        label_words = {'RECEIPT', 'INVOICE', 'BILL', 'DOCUMENT', 'DOC', 'REFERENCE', 
                      'REF', 'ORDER', 'TRANSACTION', 'TRANS', 'TXN', 'PAYMENT', 'ID'}
        if value.upper().strip() in label_words:
            return {
                "is_valid": False,
                "category": "label_word",
                "confidence": 0.0
            }
        
        # Check if it's a date pattern
        from modules.ocr_verification import is_date_pattern
        if is_date_pattern(value):
            return {
                "is_valid": False,
                "category": "date_pattern",
                "confidence": 0.0
            }
        
        # Valid if it has alphanumeric mix or is long
        has_letters = bool(re.search(r'[A-Z]', value.upper()))
        has_numbers = bool(re.search(r'[0-9]', value))
        is_long = len(value) >= 8
        
        if (has_letters and has_numbers) or is_long:
            return {
                "is_valid": True,
                "category": "transaction_id",
                "confidence": 0.8 if (has_letters and has_numbers) else 0.6
            }
        
        return {
            "is_valid": False,
            "category": "unknown",
            "confidence": 0.3
        }
    
    return {
        "is_valid": True,
        "category": field_type,
        "confidence": 0.7
    }


def _sanitize_ocr_text(text: str) -> str:
    """
    Sanitize OCR text to avoid triggering safety filters.
    Removes potentially problematic characters while preserving structure.
    """
    # Remove non-ASCII characters that might trigger filters
    sanitized = ''.join(char if ord(char) < 128 else ' ' for char in text)
    
    # Remove excessive special characters
    sanitized = re.sub(r'[^\w\s\.\,\:\-\/\(\)\$\€\£\@\#\%]', ' ', sanitized)
    
    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    return sanitized.strip()


def extract_with_llm_sanitized(sanitized_text: str) -> Dict[str, Any]:
    """
    Simplified version that uses already-sanitized text.
    This is called when the first attempt fails due to safety filters or MAX_TOKENS.
    """
    if not GEMINI_AVAILABLE or not settings.GEMINI_API_KEY:
        return {
            "transaction_id": None,
            "amount": None,
            "date": None,
            "confidence": {},
            "error": "Cannot retry - Gemini not available"
        }
    
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.LLM_MODEL or "gemini-2.5-pro")
        
        prompt = f"""Extract from OCR:

{sanitized_text[:1500]}

Find: transaction_id (like "TD01167104"), amount (total paid), date (DD/MM/YYYY)

Also provide a short text-based authenticity_check (from OCR text/structure only).

Return JSON only using this schema:
{{"transaction_id":"X","amount":"Y","date":"Z","confidence":{{"transaction_id":0.0,"amount":0.0,"date":0.0}},"authenticity_check":{{"manipulated":false,"reason":"...","confidence":0.0,"areas_of_concern":[]}}}}"""

        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, 
                max_output_tokens=2048  # INCREASED from 800
            ),
            safety_settings=safety_settings
        )
        
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            logger.error("[LLM] Sanitized attempt also failed - no content")
            return {
                "transaction_id": None,
                "amount": None,
                "date": None,
                "confidence": {},
                "error": "Even sanitized text was blocked"
            }
        
        candidate = response.candidates[0]
        
        if candidate.finish_reason == 2:
            logger.error("[LLM] MAX_TOKENS hit even with sanitized text and 2048 tokens")
           
            try:
                partial_text = response.text.strip()
                logger.debug(f"[LLM] Partial response: {partial_text}")
              
                if '{' in partial_text:
                    partial_text = partial_text[partial_text.index('{'):]
                   
                    if partial_text.count('{') > partial_text.count('}'):
                        partial_text += '}'
                    result = json.loads(partial_text)
                    logger.warning("[LLM] Recovered partial JSON from truncated response")
                    
                    # Normalize and return similar schema
                    authenticity = result.get("authenticity_check", {}) or {}
                    try:
                        auth_conf = float(authenticity.get("confidence", 0.0))
                        if auth_conf > 1.0:
                            auth_conf = auth_conf / 100.0
                        auth_conf = max(0.0, min(1.0, auth_conf))
                    except Exception:
                        auth_conf = 0.0
                    authenticity["confidence"] = auth_conf
                    
                    return {
                        "transaction_id": result.get("transaction_id"),
                        "amount": result.get("amount"),
                        "date": result.get("date"),
                        "confidence": result.get("confidence", {}),
                        "authenticity_check": authenticity,
                        "corrected": True,
                        "extraction_method": "llm_sanitized_partial"
                    }
            except:
                pass
            
            return {
                "transaction_id": None,
                "amount": None,
                "date": None,
                "confidence": {},
                "error": "Response truncated even after optimization"
            }
        
        response_text = response.text.strip()
        logger.debug(f"[LLM] Sanitized response: {response_text[:200]}...")
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        result = json.loads(response_text)
        
        # Normalize authenticity confidence
        authenticity = result.get("authenticity_check", {}) or {}
        try:
            auth_conf = float(authenticity.get("confidence", 0.0))
            if auth_conf > 1.0:
                auth_conf = auth_conf / 100.0
            auth_conf = max(0.0, min(1.0, auth_conf))
        except Exception:
            auth_conf = 0.0
        authenticity["confidence"] = auth_conf
        
        logger.info(f"[LLM] ✅ Extracted (sanitized): transaction_id={result.get('transaction_id')}, amount={result.get('amount')}, date={result.get('date')}")
        
        return {
            "transaction_id": result.get("transaction_id"),
            "amount": result.get("amount"),
            "date": result.get("date"),
            "confidence": result.get("confidence", {}),
            "explanations": result.get("explanations", {}),
            "authenticity_check": authenticity,
            "corrected": True,
            "extraction_method": "llm_sanitized"
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"[LLM] JSON parsing error in sanitized extraction: {e}")
        try:
            logger.debug(f"[LLM] Failed to parse: {response_text[:500]}")
        except:
            pass
        return {
            "transaction_id": None,
            "amount": None,
            "date": None,
            "confidence": {},
            "error": f"JSON parsing failed: {str(e)}"
        }
    except Exception as e:
        logger.error(f"[LLM] Sanitized extraction also failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "transaction_id": None,
            "amount": None,
            "date": None,
            "confidence": {},
            "error": f"Sanitized extraction failed: {str(e)}"
        }
