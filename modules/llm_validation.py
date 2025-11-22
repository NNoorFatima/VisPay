"""
LLM-based extraction and categorization for receipt fields.
Primary method: Extract all OCR text, feed to LLM for categorization.
Uses Google Gemini API (free tier).
"""
import json
import re
from typing import Dict, Optional, Any
import logging
from config import settings

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI (Gemini) not available. Install with: pip install google-generativeai")


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
        
        prompt = f"""Extract receipt data from this OCR text:

{raw_text[:2000]}

Extract these 3 fields:
1. transaction_id: Document/Invoice/Receipt number (alphanumeric code like "TD01167104", NOT words like "RECEIPT" or "INVOICE")
2. amount: Total amount paid (largest monetary value)
3. date: Transaction date (DD/MM/YYYY format)

Respond with ONLY this JSON (no markdown, no explanation):
{{"transaction_id":"value_or_null","amount":"value_or_null","date":"value_or_null","confidence":{{"transaction_id":0.0,"amount":0.0,"date":0.0}}}}"""

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
        
        logger.info(f"[LLM] Extracted: transaction_id={result.get('transaction_id')}, amount={result.get('amount')}, date={result.get('date')}")
        
        return {
            "transaction_id": result.get("transaction_id"),
            "amount": result.get("amount"),
            "date": result.get("date"),
            "confidence": result.get("confidence", {}),
            "explanations": result.get("explanations", {}),
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

JSON only (no explanation):
{{"transaction_id":"X","amount":"Y","date":"Z","confidence":{{"transaction_id":0.0,"amount":0.0,"date":0.0}}}}"""

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
                    return {
                        "transaction_id": result.get("transaction_id"),
                        "amount": result.get("amount"),
                        "date": result.get("date"),
                        "confidence": result.get("confidence", {}),
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
        
        logger.info(f"[LLM] ✅ Extracted (sanitized): transaction_id={result.get('transaction_id')}, amount={result.get('amount')}, date={result.get('date')}")
        
        return {
            "transaction_id": result.get("transaction_id"),
            "amount": result.get("amount"),
            "date": result.get("date"),
            "confidence": result.get("confidence", {}),
            "explanations": result.get("explanations", {}),
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