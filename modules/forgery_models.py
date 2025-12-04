"""
Deep Learning-based Forgery Detection Module - FIXED VERSION.
Integrates TruFor and ManTraNet models with proper error handling.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL LOADING AND INITIALIZATION
# ============================================================================

class DeepForgeryDetector:
    """
    Wrapper class for TruFor and ManTraNet models.
    Handles model loading, preprocessing, and inference with proper error handling.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize forgery detection models.
        
        Args:
            device: Device to run models on ("cpu", "cuda", or "auto")
        """
        self.device = self._get_device(device)
        self.trufor_model = None
        self.mantranet_model = None
        self.trufor_available = False
        self.mantranet_available = False
        
        logger.info(f"[DEEP_FORGERY] Initializing on device: {self.device}")
        
        # Try to load models (gracefully handle failures)
        self._load_trufor()
        self._load_mantranet()
        
        # Log final status
        if not self.trufor_available and not self.mantranet_available:
            logger.warning("[DEEP_FORGERY] No deep learning models available. "
                          "Pipeline will use traditional forensic methods only.")
        elif self.trufor_available and not self.mantranet_available:
            logger.info("[DEEP_FORGERY] TruFor loaded, ManTraNet unavailable")
        elif not self.trufor_available and self.mantranet_available:
            logger.info("[DEEP_FORGERY] ManTraNet loaded, TruFor unavailable")
        else:
            logger.info("[DEEP_FORGERY] Both TruFor and ManTraNet loaded successfully")
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate PyTorch device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_trufor(self):
        """Load TruFor model with comprehensive error handling."""
        try:
            # Check if TruFor module exists
            trufor_path = Path("models/TruFor")
            if not trufor_path.exists():
                logger.warning(f"[DEEP_FORGERY] TruFor directory not found at {trufor_path}")
                logger.info("[DEEP_FORGERY] Clone from: git clone https://github.com/grip-unina/TruFor.git models/TruFor")
                return
            
            # Try to import TruFor
            try:
                import sys
                if str(trufor_path) not in sys.path:
                    sys.path.insert(0, str(trufor_path))
                
                # Try different import paths
                try:
                    from test_docker.src.models.trufor import TruFor
                except ImportError:
                    from models.trufor import TruFor
                
                self.trufor_available = True
                
            except ImportError as e:
                logger.warning(f"[DEEP_FORGERY] TruFor module not found: {e}")
                logger.info("[DEEP_FORGERY] Install TruFor:")
                logger.info("  cd models/TruFor && pip install -r requirements.txt")
                return
            
            # Check for weights
            weights_path = trufor_path / "weights" / "trufor_weights.pth"
            if not weights_path.exists():
                logger.warning(f"[DEEP_FORGERY] TruFor weights not found at {weights_path}")
                logger.info("[DEEP_FORGERY] Download weights:")
                logger.info("  wget https://www.grip.unina.it/download/trufor/trufor_weights.pth -P models/TruFor/weights/")
                self.trufor_available = False
                return
            
            # Load model
            logger.info("[DEEP_FORGERY] Loading TruFor model...")
            self.trufor_model = TruFor(version="noema")
            
            checkpoint = torch.load(str(weights_path), map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.trufor_model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    self.trufor_model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.trufor_model.load_state_dict(checkpoint)
            else:
                self.trufor_model.load_state_dict(checkpoint)
            
            self.trufor_model.to(self.device)
            self.trufor_model.eval()
            
            logger.info("[DEEP_FORGERY] TruFor model loaded successfully ✓")
            
        except Exception as e:
            logger.error(f"[DEEP_FORGERY] Failed to load TruFor: {e}")
            import traceback
            logger.debug(f"[DEEP_FORGERY] Traceback: {traceback.format_exc()}")
            self.trufor_available = False
            self.trufor_model = None
    
    def _load_mantranet(self):
        """Load ManTraNet model with comprehensive error handling."""
        try:
            # Check if ManTraNet module exists
            mantranet_path = Path("models/ManTraNet")
            if not mantranet_path.exists():
                logger.warning(f"[DEEP_FORGERY] ManTraNet directory not found at {mantranet_path}")
                logger.info("[DEEP_FORGERY] Clone from: git clone https://github.com/ISICV/ManTraNet.git models/ManTraNet")
                return
            
            # Try to import ManTraNet
            try:
                import sys
                if str(mantranet_path) not in sys.path:
                    sys.path.insert(0, str(mantranet_path))
                
                from mantranet import MANTRANET
                self.mantranet_available = True
                
            except ImportError as e:
                logger.warning(f"[DEEP_FORGERY] ManTraNet module not found: {e}")
                logger.info("[DEEP_FORGERY] Install ManTraNet:")
                logger.info("  cd models/ManTraNet && pip install -r requirements.txt")
                return
            
            # Check for weights
            weights_path = mantranet_path / "weights" / "mantranet_weights.pth"
            if not weights_path.exists():
                logger.warning(f"[DEEP_FORGERY] ManTraNet weights not found at {weights_path}")
                logger.info("[DEEP_FORGERY] Download weights:")
                logger.info("  Visit: https://github.com/ISICV/ManTraNet/releases")
                self.mantranet_available = False
                return
            
            # Load model
            logger.info("[DEEP_FORGERY] Loading ManTraNet model...")
            self.mantranet_model = MANTRANET()
            
            checkpoint = torch.load(str(weights_path), map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    self.mantranet_model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint:
                    self.mantranet_model.load_state_dict(checkpoint['model'])
                else:
                    self.mantranet_model.load_state_dict(checkpoint)
            else:
                self.mantranet_model.load_state_dict(checkpoint)
            
            self.mantranet_model.to(self.device)
            self.mantranet_model.eval()
            
            logger.info("[DEEP_FORGERY] ManTraNet model loaded successfully ✓")
            
        except Exception as e:
            logger.error(f"[DEEP_FORGERY] Failed to load ManTraNet: {e}")
            import traceback
            logger.debug(f"[DEEP_FORGERY] Traceback: {traceback.format_exc()}")
            self.mantranet_available = False
            self.mantranet_model = None
    
    def preprocess_for_trufor(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for TruFor model."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512))
        pil_img = Image.fromarray(resized)
        tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)
    
    def preprocess_for_mantranet(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for ManTraNet model."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (256, 256))
        tensor = torch.from_numpy(resized).float()
        tensor = (tensor / 127.5) - 1.0
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)
    
    def run_trufor(self, img: np.ndarray) -> Dict[str, Any]:
        """Run TruFor forgery detection."""
        if not self.trufor_available or self.trufor_model is None:
            logger.debug("[DEEP_FORGERY] TruFor not available, skipping")
            return None  # Return None instead of error dict
        
        try:
            tensor = self.preprocess_for_trufor(img)
            
            with torch.no_grad():
                output = self.trufor_model(tensor)
            
            # TruFor outputs: dict with 'det', 'conf', 'np' keys
            if isinstance(output, dict):
                confidence_map = output['conf'].cpu().numpy()[0, 0]
                detection_mask = output['det'].cpu().numpy()[0, 0]
            else:
                # Handle tuple output
                confidence_map = output[0].cpu().numpy()[0]
                detection_mask = output[1].cpu().numpy()[0]
            
            manipulation_ratio = np.mean(detection_mask > 0.5)
            authenticity_score = 1.0 - manipulation_ratio
            avg_confidence = float(np.mean(confidence_map))
            
            result = {
                "authenticity_score": float(authenticity_score),
                "manipulation_ratio": float(manipulation_ratio),
                "avg_confidence": avg_confidence,
                "manipulated_pixels": int(np.sum(detection_mask > 0.5)),
                "total_pixels": detection_mask.size,
                "has_manipulation": manipulation_ratio > 0.15,
                "detection_mask": detection_mask,
                "confidence_map": confidence_map
            }
            
            logger.info(f"[DEEP_FORGERY] TruFor: authenticity={authenticity_score:.3f}, "
                       f"manipulation_ratio={manipulation_ratio:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"[DEEP_FORGERY] TruFor inference failed: {e}")
            return None
    
    def run_mantranet(self, img: np.ndarray) -> Dict[str, Any]:
        """Run ManTraNet forgery detection."""
        if not self.mantranet_available or self.mantranet_model is None:
            logger.debug("[DEEP_FORGERY] ManTraNet not available, skipping")
            return None  # Return None instead of error dict
        
        try:
            tensor = self.preprocess_for_mantranet(img)
            
            with torch.no_grad():
                output = self.mantranet_model(tensor)
            
            prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
            manipulation_ratio = np.mean(prob_map > 0.5)
            authenticity_score = 1.0 - manipulation_ratio
            manipulation_type = self._classify_manipulation_type(prob_map)
            
            result = {
                "authenticity_score": float(authenticity_score),
                "manipulation_ratio": float(manipulation_ratio),
                "manipulation_type": manipulation_type,
                "manipulated_pixels": int(np.sum(prob_map > 0.5)),
                "total_pixels": prob_map.size,
                "has_manipulation": manipulation_ratio > 0.15,
                "probability_map": prob_map
            }
            
            logger.info(f"[DEEP_FORGERY] ManTraNet: authenticity={authenticity_score:.3f}, "
                       f"type={manipulation_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"[DEEP_FORGERY] ManTraNet inference failed: {e}")
            return None
    
    def _classify_manipulation_type(self, prob_map: np.ndarray) -> str:
        """Classify type of manipulation based on probability map patterns."""
        if np.max(prob_map) < 0.3:
            return "none"
        
        high_prob_regions = prob_map > 0.5
        
        if np.sum(high_prob_regions) == 0:
            return "none"
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            high_prob_regions.astype(np.uint8)
        )
        
        if num_labels <= 2:
            return "localized_edit"
        elif num_labels <= 5:
            return "multiple_edits"
        else:
            return "extensive_manipulation"
    
    def detect_forgery(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Run both TruFor and ManTraNet and combine results.
        FIXED: Properly handles None returns from models.
        """
        results = {
            "trufor": None,
            "mantranet": None,
            "combined_score": 0.5,
            "recommendation": "UNKNOWN",
            "models_available": [],
            "error": None
        }
        
        # Check if any models are available
        if not self.trufor_available and not self.mantranet_available:
            results["error"] = "No deep learning models available"
            logger.warning("[DEEP_FORGERY] No models available for inference")
            return results
        
        # Run TruFor
        if self.trufor_available and self.trufor_model is not None:
            trufor_result = self.run_trufor(img)
            if trufor_result is not None:  # Check for None
                results["trufor"] = trufor_result
                results["models_available"].append("trufor")
        
        # Run ManTraNet
        if self.mantranet_available and self.mantranet_model is not None:
            mantranet_result = self.run_mantranet(img)
            if mantranet_result is not None:  # Check for None
                results["mantranet"] = mantranet_result
                results["models_available"].append("mantranet")
        
        # Combine scores (FIXED: check if results exist)
        scores = []
        weights = []
        
        if results["trufor"] is not None and "authenticity_score" in results["trufor"]:
            scores.append(results["trufor"]["authenticity_score"])
            weights.append(0.55)
        
        if results["mantranet"] is not None and "authenticity_score" in results["mantranet"]:
            scores.append(results["mantranet"]["authenticity_score"])
            weights.append(0.45)
        
        if scores:
            combined_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            results["combined_score"] = float(combined_score)
            
            # Determine recommendation
            if combined_score > 0.85:
                results["recommendation"] = "AUTHENTIC"
            elif combined_score > 0.70:
                results["recommendation"] = "LIKELY_AUTHENTIC"
            elif combined_score > 0.40:
                results["recommendation"] = "SUSPICIOUS"
            else:
                results["recommendation"] = "LIKELY_FORGED"
            
            # Clear error if we got results
            results["error"] = None
        else:
            # No models produced results
            results["error"] = "Model inference failed"
            logger.warning("[DEEP_FORGERY] No models produced valid results")
        
        logger.info(f"[DEEP_FORGERY] Combined Score: {results['combined_score']:.3f} - "
                   f"{results['recommendation']}")
        
        return results


# ============================================================================
# INTEGRATION FUNCTION FOR EXISTING PIPELINE
# ============================================================================

_detector_instance = None

def get_detector() -> DeepForgeryDetector:
    """Get or create global detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DeepForgeryDetector()
    return _detector_instance


def check_deep_forgery(img: np.ndarray) -> Dict[str, Any]:
    """
    Main function to integrate into existing pipeline.
    FIXED: Properly handles all error cases.
    """
    try:
        detector = get_detector()
        
        # Check if any models are available
        if not detector.trufor_available and not detector.mantranet_available:
            return {
                "deep_learning_check": False,
                "models_used": [],
                "authenticity_score": 0.5,
                "recommendation": "UNKNOWN",
                "error": "No deep learning models available",
                "has_manipulation": False
            }
        
        results = detector.detect_forgery(img)
        
        # Handle case where models failed
        if results.get("error") and not results.get("models_available"):
            return {
                "deep_learning_check": False,
                "models_used": [],
                "authenticity_score": 0.5,
                "recommendation": "UNKNOWN",
                "error": results.get("error"),
                "has_manipulation": False
            }
        
        # Format results for pipeline integration
        return {
            "deep_learning_check": True,
            "models_used": results["models_available"],
            "authenticity_score": results["combined_score"],
            "recommendation": results["recommendation"],
            "trufor_details": results.get("trufor", {}),
            "mantranet_details": results.get("mantranet", {}),
            "has_manipulation": (
                (results.get("trufor", {}) or {}).get("has_manipulation", False) or
                (results.get("mantranet", {}) or {}).get("has_manipulation", False)
            ),
            "error": results.get("error")
        }
        
    except Exception as e:
        logger.error(f"[DEEP_FORGERY] Deep forgery check failed: {e}")
        import traceback
        logger.debug(f"[DEEP_FORGERY] Traceback: {traceback.format_exc()}")
        return {
            "deep_learning_check": False,
            "error": str(e),
            "authenticity_score": 0.5,
            "recommendation": "ERROR",
            "models_used": [],
            "has_manipulation": False
        }


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_manipulation_mask(
    img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Overlay manipulation mask on original image for visualization."""
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    overlay = img.copy()
    red_mask = (mask_resized > 0.5).astype(np.uint8)
    overlay[red_mask == 1] = [0, 0, 255]
    result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return result


def create_heatmap(
    prob_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Create colored heatmap from probability map."""
    heatmap = (prob_map * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, colormap)
    return colored_heatmap