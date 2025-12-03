"""
Receipt Image Authenticity and Manipulation Detection Module.
Implements 7 forensic techniques to detect if a receipt has been digitally altered.

Approaches:
1. Metadata Analysis (EXIF/IPTC)
2. Pixel-Level Forensics (Copy-Move Detection)
3. Compression Artifact Analysis
4. Font Consistency Check
5. Semantic Consistency Check (LLM-based)
6. Noise Pattern Analysis
7. Optical/Perspective Distortion Check
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. EXIF extraction will be skipped.")

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False
    logger.warning("piexif not available. EXIF parsing will be limited.")


# ============================================================================
# 1. METADATA ANALYSIS (EXIF/IPTC)
# ============================================================================

def extract_exif_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract EXIF metadata from image to detect editing software and timestamps.
    
    Returns:
        Dictionary with EXIF data including creation date, modification date,
        camera info, and editing software markers.
    """
    if not PIL_AVAILABLE:
        logger.warning("[AUTHENTICITY] PIL not available for EXIF extraction")
        return {"error": "PIL not available"}
    
    try:
        image = Image.open(image_path)
        exif_data = image._getexif() if hasattr(image, '_getexif') else None
        
        if not exif_data:
            logger.debug("[AUTHENTICITY] No EXIF data found in image")
            return {"exif_present": False, "note": "No EXIF data"}
        
        exif_dict = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            exif_dict[tag] = str(value)[:100]  # Truncate long values
        
        logger.debug(f"[AUTHENTICITY] Extracted EXIF: {list(exif_dict.keys())}")
        
        return {
            "exif_present": True,
            "exif_data": exif_dict,
            "has_camera_info": any(k in exif_dict for k in ["Model", "Make", "LensModel"]),
            "has_software_marker": any(k in exif_dict for k in ["Software", "ProcessingSoftware"]),
            "software": exif_dict.get("Software", "Unknown")
        }
    
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Error extracting EXIF: {e}")
        return {"error": str(e)}


def check_metadata_integrity(image_path: str) -> float:
    """
    Analyze metadata for signs of tampering.
    
    Score interpretation:
        1.0 = Metadata present and consistent (more authentic)
        0.5 = No metadata (neutral - digital receipts often have no EXIF)
        0.0 = Suspicious metadata patterns (potential forgery)
    
    Returns:
        Float score 0.0-1.0
    """
    metadata = extract_exif_metadata(image_path)
    
    if "error" in metadata or not metadata.get("exif_present"):
        # Digital receipts (PDFs exported as PNG/JPG) often have no EXIF
        logger.debug("[AUTHENTICITY] No EXIF data - typical for digital receipts (neutral)")
        return 0.5
    
    exif_dict = metadata.get("exif_data", {})
    
    # Check for suspicious patterns
    software = metadata.get("software", "").lower()
    suspicious_editors = ["photoshop", "gimp", "photopea", "pixlr", "canva"]
    
    for editor in suspicious_editors:
        if editor in software:
            logger.warning(f"[AUTHENTICITY] Suspicious editor detected: {editor}")
            return 0.2  # Low score: image likely edited with graphics software
    
    # Check for consistency between creation and modification times
    if "DateTime" in exif_dict and "DateTimeOriginal" in exif_dict:
        if exif_dict["DateTime"] != exif_dict["DateTimeOriginal"]:
            logger.debug("[AUTHENTICITY] Creation and modification dates differ (possible edit)")
            return 0.6
    
    logger.info(f"[AUTHENTICITY] Metadata appears authentic (software: {software})")
    return 0.8


# ============================================================================
# 2. PIXEL-LEVEL FORENSICS (COPY-MOVE DETECTION)
# ============================================================================

def compute_block_hash(block: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute a simple hash signature for an image block.
    Uses DCT (Discrete Cosine Transform) coefficients.
    
    Args:
        block: Image block (typically 16x16 or 32x32)
        hash_size: Size of hash output (default 8x8 = 64 bits)
    
    Returns:
        String representation of hash
    """
    try:
        # Resize block to hash_size for consistent hashing
        resized = cv2.resize(block, (hash_size, hash_size))
        
        # Convert to grayscale if needed
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Compute DCT
        dct = cv2.dct(np.float32(resized) / 255.0)
        
        # Use low-frequency coefficients for hash
        avg_val = np.mean(dct[:hash_size//2, :hash_size//2])
        
        # Convert to binary hash
        hash_bits = ''.join(['1' if dct[i, j] > avg_val else '0' 
                            for i in range(hash_size) for j in range(hash_size)])
        return hash_bits
    
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Error computing block hash: {e}")
        return ""


def detect_copy_move_forgery(img: np.ndarray, block_size: int = 32, threshold: float = 0.95) -> float:
    """
    Detect copy-move forgery using block matching.
    
    Divides image into blocks, computes hashes, and searches for duplicates.
    High similarity between distant blocks indicates potential copy-paste.
    
    Args:
        img: Image as numpy array
        block_size: Size of blocks to analyze (default 32x32)
        threshold: Similarity threshold for flagging (0.0-1.0)
    
    Returns:
        Float score 0.0-1.0 where 0.0 = likely forged, 1.0 = authentic
    """
    try:
        if img is None or img.size == 0:
            return 0.5
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        height, width = gray.shape
        
        # Extract blocks and compute hashes
        block_hashes = defaultdict(list)  # hash -> list of (x, y) coordinates
        blocks_processed = 0

        # Heuristic: skip near-uniform/low-variance blocks because they produce
        # identical hashes across the image (background, margins) and cause
        # massive false-positive duplicate counts.
        VAR_THRESHOLD = 30.0  # tuned; blocks with variance below this are ignored
        MAX_LOCS_PER_HASH = 50  # if a hash appears in >50 blocks, treat as background

        # iterate blocks across full image; include last partial blocks
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                y_end = min(y + block_size, height)
                x_end = min(x + block_size, width)
                block = gray[y:y_end, x:x_end]
                if block.size == 0:
                    continue

                # compute variance and skip very-uniform blocks
                block_var = float(np.var(block))
                if block_var < VAR_THRESHOLD:
                    # skip background/margin blocks
                    logger.debug(f"[AUTHENTICITY] Skipping low-variance block at ({x},{y}) var={block_var:.2f}")
                    continue

                blocks_processed += 1
                block_hash = compute_block_hash(block)

                if not block_hash:
                    continue
                # store coordinates and small diagnostic (we don't need variance later)
                block_hashes[block_hash].append((x, y))

        # Count suspicious duplicates (same hash at different, non-adjacent locations)
        duplicate_pairs = 0
        # minimum distance (pixels) between blocks to consider them non-local duplicates
        min_distance = max(block_size * 2, 100)

        for hash_val, locations in block_hashes.items():
            loc_count = len(locations)
            # If a hash occurs excessively across the image it's likely a background
            # pattern (e.g., large white margins) despite our variance check; skip it.
            if loc_count <= 1 or loc_count > MAX_LOCS_PER_HASH:
                if loc_count > MAX_LOCS_PER_HASH:
                    logger.debug(f"[AUTHENTICITY] Ignoring hash {hash_val} with {loc_count} locations (likely background)")
                continue

            # compute number of distant pairs for this hash
            locs = locations
            pair_count = 0
            for i in range(len(locs)):
                x1, y1 = locs[i]
                for j in range(i + 1, len(locs)):
                    x2, y2 = locs[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist >= min_distance:
                        pair_count += 1

            if pair_count > 0:
                # cap pair_count to avoid single-hash quadratic blowups
                capped_pairs = min(pair_count, loc_count * 5)
                duplicate_pairs += capped_pairs
                logger.debug(f"[AUTHENTICITY] Found {pair_count} distant duplicate pairs for hash {hash_val} (capped to {capped_pairs})")

        total_blocks = blocks_processed if blocks_processed > 0 else 0

        if total_blocks == 0:
            return 0.5

        # Normalize by number of processed blocks to get a forgery-like ratio
        forgery_ratio = duplicate_pairs / float(total_blocks)

        logger.debug(f"[AUTHENTICITY] duplicate_pairs={duplicate_pairs}, blocks_processed={blocks_processed}, forgery_ratio={forgery_ratio:.4f}")

        # Thresholds calibrated for distant-pair based metric
        if forgery_ratio > 0.06:  # >6% distant-pair rate = suspicious
            logger.warning(f"[AUTHENTICITY] High copy-move forgery ratio (distant pairs): {forgery_ratio:.2%}")
            return 0.15
        elif forgery_ratio > 0.02:
            logger.info(f"[AUTHENTICITY] Moderate copy-move ratio (distant pairs): {forgery_ratio:.2%}")
            return 0.55
        else:
            logger.debug(f"[AUTHENTICITY] Low copy-move ratio (distant pairs): {forgery_ratio:.2%}")
            return 0.92
    
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Error in copy-move detection: {e}")
        return 0.5


# ============================================================================
# 3. COMPRESSION ARTIFACT ANALYSIS
# ============================================================================

def analyze_jpeg_artifacts(image_path: str) -> float:
    """
    Detect JPEG re-compression artifacts (sign of tampering).
    
    Multiple save cycles create unnatural compression patterns.
    
    Returns:
        Float score 0.0-1.0 where 0.0 = likely re-compressed, 1.0 = single save
    """
    try:
        if not image_path.lower().endswith(('.jpg', '.jpeg')):
            logger.debug("[AUTHENTICITY] Not a JPEG - skipping artifact analysis")
            return 0.7
        
        img = cv2.imread(image_path)
        
        # Compute DCT coefficients (JPEG uses DCT)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute 2D DCT
        dct = cv2.dct(np.float32(gray) / 255.0)
        
        # Analyze coefficient distribution
        # Re-compressed images show unnatural patterns
        coeff_histogram = cv2.calcHist([dct], [0], None, [256], [0, 1])
        
        # Look for sharp peaks (sign of re-compression)
        peak_count = np.sum(coeff_histogram > np.mean(coeff_histogram) * 2)
        peak_ratio = peak_count / len(coeff_histogram)
        
        if peak_ratio > 0.3:
            logger.warning(f"[AUTHENTICITY] High compression artifacts detected (ratio: {peak_ratio:.2%})")
            return 0.4
        elif peak_ratio > 0.15:
            logger.info(f"[AUTHENTICITY] Moderate compression artifacts (ratio: {peak_ratio:.2%})")
            return 0.7
        else:
            logger.debug("[AUTHENTICITY] Low compression artifacts - likely single save")
            return 0.9
    
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Error analyzing JPEG artifacts: {e}")
        return 0.5


def analyze_compression_artifacts(image_path: str) -> Dict[str, Any]:
    """
    Comprehensive compression artifact analysis.
    """
    jpeg_score = analyze_jpeg_artifacts(image_path)
    
    return {
        "jpeg_artifact_score": jpeg_score,
        "overall_compression_score": jpeg_score,
        "interpretation": (
            "Single save (likely authentic)" if jpeg_score > 0.8 else
            "Moderate re-compression (suspicious)" if jpeg_score > 0.5 else
            "Multiple re-compression (likely forged)"
        )
    }


# ============================================================================
# 4. FONT CONSISTENCY CHECK
# ============================================================================

def check_ocr_confidence_consistency(ocr_results: List[Tuple[str, float, Tuple]]) -> float:
    """
    Analyze OCR confidence scores to detect edited text.
    
    EasyOCR provides confidence per line. Sudden drops indicate potential edits.
    
    Args:
        ocr_results: List of (text, confidence, bbox) from EasyOCR
    
    Returns:
        Float score 0.0-1.0 where 1.0 = consistent confidence, 0.0 = suspicious variance
    """
    try:
        if not ocr_results or len(ocr_results) < 2:
            return 0.8
        
        confidences = []
        for item in ocr_results:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                conf = float(item[1])
                confidences.append(conf)
        
        if not confidences:
            return 0.8
        
        # Analyze confidence distribution
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        # High variance in confidence = potential text alterations
        cv = std_conf / mean_conf if mean_conf > 0 else 0
        
        if cv > 0.3:  # Coefficient of variation > 30% = suspicious
            logger.warning(f"[AUTHENTICITY] High OCR confidence variance (CV: {cv:.2f})")
            min_conf = min(confidences)
            max_conf = max(confidences)
            logger.debug(f"[AUTHENTICITY] Confidence range: {min_conf:.2f} - {max_conf:.2f}")
            return 0.4
        elif cv > 0.15:
            logger.info(f"[AUTHENTICITY] Moderate OCR variance (CV: {cv:.2f})")
            return 0.7
        else:
            logger.debug(f"[AUTHENTICITY] Consistent OCR confidence (CV: {cv:.2f})")
            return 0.9
    
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Error checking OCR consistency: {e}")
        return 0.5


def analyze_font_consistency(img: np.ndarray, ocr_results: List[Any]) -> Dict[str, Any]:
    """
    Analyze font consistency across detected text regions.
    """
    confidence_score = 0.8
    if ocr_results:
        confidence_score = check_ocr_confidence_consistency(ocr_results)
    
    return {
        "ocr_confidence_consistency": confidence_score,
        "interpretation": (
            "Consistent font/confidence" if confidence_score > 0.8 else
            "Variable confidence (possible edits)" if confidence_score > 0.5 else
            "High variance (likely edited)"
        )
    }


# ============================================================================
# 5. SEMANTIC CONSISTENCY CHECK (LLM-based)
# ============================================================================

def check_semantic_consistency_with_llm(raw_text: str) -> Dict[str, Any]:
    """
    Use Gemini LLM to check logical consistency in receipt data.
    
    Detects:
        - Amount math inconsistencies
        - Date format conflicts
        - Mismatched line items and totals
        - Logical contradictions
    
    This is called from llm_validation.extract_with_llm with enhanced prompt.
    
    Returns:
        Dictionary with authenticity indicators
    """
    try:
        import google.generativeai as genai
        from config import settings
        
        if not settings.GEMINI_API_KEY:
            logger.warning("[AUTHENTICITY] GEMINI_API_KEY not configured - skipping semantic check")
            return {"error": "GEMINI_API_KEY not configured", "authenticity_confidence": 0.5}
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.LLM_MODEL or "gemini-2.5-pro")
        
        semantic_prompt = f"""Analyze receipt for forgery signs. Be brief.

Text:
{raw_text[:800]}

Check ONLY these 3 things:
1. Look at the spaces in the receipt does it all look consistent?
2. Are date formats consistent? (true/false)
3. Does layout look typical? (true/false)

Output ONLY valid JSON (no text before or after):
{{"amounts_consistent": true, "date_format_consistent": true, "layout_typical": true, "authenticity_confidence": 0.75, "recommendation": "authentic"}}"""
        
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        response = model.generate_content(
            semantic_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=200  # REDUCED from 1024 to avoid truncation
            ),
            safety_settings=safety_settings
        )
        
        # Check finish_reason before accessing response.text
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            # Handle truncation (finish_reason == 2 = MAX_TOKENS)
            if candidate.finish_reason == 2:
                logger.warning("[AUTHENTICITY] LLM response truncated (MAX_TOKENS) - returning neutral score")
                return {
                    "authenticity_confidence": 0.5,
                    "recommendation": "review",
                    "amounts_consistent": True,
                    "date_format_consistent": True,
                    "layout_typical": True,
                    "error": "Response truncated"
                }
            
            # Handle safety blocks (finish_reason == 3 = SAFETY)
            if candidate.finish_reason == 3:
                logger.warning("[AUTHENTICITY] LLM response blocked by safety filters")
                return {
                    "authenticity_confidence": 0.5,
                    "recommendation": "review",
                    "error": "Safety filter triggered"
                }
            
            # Check if content is available before accessing response.text
            if not candidate.content or not candidate.content.parts:
                logger.warning("[AUTHENTICITY] LLM returned no content")
                return {
                    "authenticity_confidence": 0.5,
                    "recommendation": "review",
                    "error": "No response content"
                }
            
            # Now safe to access response.text
            response_text = response.text.strip()
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    
                    # Ensure required fields exist
                    result.setdefault("authenticity_confidence", 0.5)
                    result.setdefault("recommendation", "review")
                    result.setdefault("amounts_consistent", True)
                    result.setdefault("date_format_consistent", True)
                    result.setdefault("layout_typical", True)
                    
                    logger.info(f"[AUTHENTICITY] LLM semantic check: {result.get('recommendation')} "
                              f"(confidence: {result.get('authenticity_confidence')})")
                    return result
                except json.JSONDecodeError as je:
                    logger.warning(f"[AUTHENTICITY] Failed to parse JSON: {je}")
                    return {
                        "authenticity_confidence": 0.5,
                        "recommendation": "review",
                        "error": f"JSON parse error: {str(je)}"
                    }
            else:
                logger.warning(f"[AUTHENTICITY] No JSON found in response: {response_text[:100]}")
                return {
                    "authenticity_confidence": 0.5,
                    "recommendation": "review",
                    "error": "No JSON in response"
                }
        else:
            logger.warning("[AUTHENTICITY] No candidates returned from Gemini API")
            return {
                "authenticity_confidence": 0.5,
                "recommendation": "review",
                "error": "No API response"
            }
    
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Error in semantic check: {e}")
        import traceback
        logger.debug(f"[AUTHENTICITY] Traceback: {traceback.format_exc()}")
        return {
            "authenticity_confidence": 0.5,
            "recommendation": "review",
            "error": str(e)
        }


# ============================================================================
# 6. NOISE PATTERN ANALYSIS
# ============================================================================

def analyze_noise_distribution(gray: np.ndarray, regions: int = 4) -> Dict[str, float]:
    """
    Analyze noise distribution across image regions.
    
    Uniform noise = authentic camera/scan capture.
    Patchy noise = possible edited regions.
    
    Args:
        gray: Grayscale image
        regions: Number of regions to divide image into (default 4 = 2x2 grid)
    
    Returns:
        Dictionary with noise scores per region
    """
    try:
        height, width = gray.shape
        region_size = int(np.sqrt(regions))
        
        region_h = height // region_size
        region_w = width // region_size
        
        noise_scores = {}

        for i in range(region_size):
            for j in range(region_size):
                y_start = i * region_h
                y_end = (i + 1) * region_h if i < region_size - 1 else height
                x_start = j * region_w
                x_end = (j + 1) * region_w if j < region_size - 1 else width

                region = gray[y_start:y_end, x_start:x_end]

                # Apply bilateral filter
                smoothed = cv2.bilateralFilter(region, 9, 75, 75)
                diff = cv2.absdiff(region, smoothed)
                noise_var = float(np.var(diff))
                # normalize noise_var into a 0-1 heuristic
                noise_score = noise_var / (noise_var + 1.0)
                region_key = f"region_{i}_{j}"
                noise_scores[region_key] = noise_score
                logger.debug(f"[AUTHENTICITY] {region_key} noise score: {noise_score:.4f}")

        # Calculate coefficient of variation across regions
        noise_values = list(noise_scores.values())
        if len(noise_values) > 1:
            mean_noise = float(np.mean(noise_values))
            std_noise = float(np.std(noise_values))
            cv_noise = std_noise / mean_noise if mean_noise > 0 else 0

            # uniformity_score: higher means more uniform noise => more likely authentic
            uniformity_score = max(0.0, 1.0 - cv_noise)
            noise_scores["cv_noise"] = cv_noise
            noise_scores["uniformity_score"] = round(float(uniformity_score), 3)
        else:
            # Not enough regions; return neutral uniformity
            noise_scores["cv_noise"] = 0.0
            noise_scores["uniformity_score"] = 0.7

        return noise_scores
    
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Error analyzing noise: {e}")
        return {"error": str(e)}


# ============================================================================
# 7. OPTICAL/PERSPECTIVE DISTORTION CHECK
# ============================================================================

def check_perspective_integrity(img: np.ndarray, edge_threshold: float = 0.02) -> float:
    """
    Detect unnatural perspective or perfectly straight edges (sign of digital creation).
    
    Real receipt photos have slight skew; pixel-perfect rectangles suggest doctoring.
    
    Args:
        img: Image as numpy array
        edge_threshold: Threshold for edge detection
    
    Returns:
        Float score 0.0-1.0 where 1.0 = natural perspective, 0.0 = suspicious perfection
    """
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is None or len(lines) == 0:
            logger.debug("[AUTHENTICITY] No clear lines detected - natural perspective")
            return 0.8
        
        # Extract angles from lines
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            angles.append(angle)
        
        # Check for perfectly horizontal/vertical lines
        # Real photos have slight variations; doctored images have exact 90°, 0°
        perfect_angles = sum(1 for angle in angles if abs(angle) < 1 or abs(angle - 90) < 1)
        perfect_ratio = perfect_angles / len(angles) if lines is not None else 0
        
        if perfect_ratio > 0.7:
            logger.warning(f"[AUTHENTICITY] Too many perfect angles ({perfect_ratio:.1%}) - likely digitally created")
            return 0.3
        elif perfect_ratio > 0.4:
            logger.info(f"[AUTHENTICITY] Moderate angle perfection ({perfect_ratio:.1%})")
            return 0.6
        else:
            logger.debug(f"[AUTHENTICITY] Natural perspective variation ({perfect_ratio:.1%})")
            return 0.9
    
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Error checking perspective: {e}")
        return 0.5


# ============================================================================
# MAIN AUTHENTICITY CHECK FUNCTION
# ============================================================================

def check_image_authenticity(
    img: np.ndarray,
    image_path: Optional[str] = None,
    ocr_results: Optional[List[Any]] = None,
    raw_ocr_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive image authenticity check using all 7 forensic methods.
    
    Args:
        img: Image as numpy array
        image_path: Path to image file (for EXIF extraction)
        ocr_results: OCR results with confidence scores
        raw_ocr_text: Raw OCR text for semantic checks
    
    Returns:
        Dictionary with authenticity scores and overall recommendation
    """
    logger.info("[AUTHENTICITY] Starting comprehensive authenticity check...")
    
    indicators = []
    
    # 1. Metadata Analysis
    if image_path:
        try:
            metadata_score = check_metadata_integrity(image_path)
            indicators.append({
                "check": "metadata_integrity",
                "score": metadata_score,
                "weight": 0.15,
                "description": "EXIF metadata analysis"
            })
            logger.debug(f"[AUTHENTICITY] Metadata score: {metadata_score:.2f}")
        except Exception as e:
            logger.warning(f"[AUTHENTICITY] Metadata check failed: {e}")
    
    # 2. Copy-Move Detection
    try:
        copy_move_score = detect_copy_move_forgery(img)
        indicators.append({
            "check": "copy_move_detection",
            "score": copy_move_score,
            "weight": 0.20,
            "description": "Pixel-level copy-move forgery detection"
        })
        logger.debug(f"[AUTHENTICITY] Copy-move score: {copy_move_score:.2f}")
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Copy-move detection failed: {e}")
    
    # 3. Compression Artifacts
    if image_path:
        try:
            comp_analysis = analyze_compression_artifacts(image_path)
            comp_score = comp_analysis.get("overall_compression_score", 0.5)
            indicators.append({
                "check": "compression_artifacts",
                "score": comp_score,
                "weight": 0.10,
                "description": "JPEG re-compression artifact detection"
            })
            logger.debug(f"[AUTHENTICITY] Compression score: {comp_score:.2f}")
        except Exception as e:
            logger.warning(f"[AUTHENTICITY] Compression analysis failed: {e}")
    
    # 4. Font Consistency
    try:
        font_analysis = analyze_font_consistency(img, ocr_results)
        font_score = font_analysis.get("ocr_confidence_consistency", 0.7)
        indicators.append({
            "check": "font_consistency",
            "score": font_score,
            "weight": 0.15,
            "description": "OCR confidence and font consistency"
        })
        logger.debug(f"[AUTHENTICITY] Font consistency score: {font_score:.2f}")
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Font consistency check failed: {e}")
    
    # 5. Semantic Consistency (LLM)
    if raw_ocr_text:
        try:
            semantic_result = check_semantic_consistency_with_llm(raw_ocr_text)
            semantic_score = semantic_result.get("authenticity_confidence", 0.5)
            indicators.append({
                "check": "semantic_consistency",
                "score": semantic_score,
                "weight": 0.20,
                "description": "LLM-based logical consistency check"
            })
            logger.debug(f"[AUTHENTICITY] Semantic score: {semantic_score:.2f}")
        except Exception as e:
            logger.warning(f"[AUTHENTICITY] Semantic check failed: {e}")
    
    # 6. Noise Distribution
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        noise_analysis = analyze_noise_distribution(gray)
        noise_score = noise_analysis.get("uniformity_score", 0.7)
        indicators.append({
            "check": "noise_distribution",
            "score": noise_score,
            "weight": 0.10,
            "description": "Regional noise pattern analysis"
        })
        logger.debug(f"[AUTHENTICITY] Noise score: {noise_score:.2f}")
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Noise analysis failed: {e}")
    
    # 7. Perspective Distortion
    try:
        perspective_score = check_perspective_integrity(img)
        indicators.append({
            "check": "perspective_distortion",
            "score": perspective_score,
            "weight": 0.10,
            "description": "Optical perspective and edge integrity"
        })
        logger.debug(f"[AUTHENTICITY] Perspective score: {perspective_score:.2f}")
    except Exception as e:
        logger.warning(f"[AUTHENTICITY] Perspective check failed: {e}")
    
    # Calculate weighted average
    if indicators:
        total_weight = sum(ind.get("weight", 0) for ind in indicators)
        weighted_sum = sum(ind["score"] * ind.get("weight", 0) for ind in indicators)
        overall_authenticity = weighted_sum / total_weight if total_weight > 0 else 0.5
    else:
        overall_authenticity = 0.5
    
    # Determine recommendation (use slightly stricter thresholds)
    if overall_authenticity > 0.80:
        recommendation = "AUTHENTIC"
        confidence_level = "HIGH"
    elif overall_authenticity > 0.65:
        recommendation = "LIKELY_AUTHENTIC"
        confidence_level = "MEDIUM"
    elif overall_authenticity > 0.45:
        recommendation = "SUSPICIOUS"
        confidence_level = "MEDIUM"
    else:
        recommendation = "LIKELY_FORGED"
        confidence_level = "HIGH"
    
    result = {
        "authenticity_score": round(overall_authenticity, 3),
        "recommendation": recommendation,
        "confidence_level": confidence_level,
        "indicators": indicators,
        "is_suspicious": overall_authenticity < 0.6,
        "details": {
            "total_checks_performed": len(indicators),
            "checks_passed": sum(1 for ind in indicators if ind["score"] > 0.7),
            "checks_failed": sum(1 for ind in indicators if ind["score"] < 0.4)
        }
    }
    
    logger.info(f"[AUTHENTICITY] Final Score: {overall_authenticity:.3f} - {recommendation}")
    
    return result
