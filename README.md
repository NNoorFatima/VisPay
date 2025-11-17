# VisPay Vision: A Dual Image Intelligence System for Payment Verification and Visual Product Search in Live Commerce
## Overview

VisPay Vision is an intelligent digital image processing solution for live commerce platforms. It addresses two major operational challenges faced by sellers:

1. Manual verification of payment receipts.
2. Difficulty identifying products from customer-uploaded images.

This project combines OCR-based payment verification and Visual Product Search using feature matching. Currently, the Visual Product Search module has been fully implemented.

## Features Implemented
### Visual Product Search Module

The Visual Product Search allows sellers to quickly identify products from customer-uploaded images during live chat or streaming sessions. The implemented pipeline includes:

1. Feature Extraction with SIFT:# VisPay Vision: A Dual Image Intelligence System for Payment Verification and Visual Product Search in Live Commerce

## Overview

VisPay Vision is an intelligent digital image processing solution for live commerce platforms. It addresses two major operational challenges faced by sellers:

1. Manual verification of payment receipts.
2. Difficulty identifying products from customer-uploaded images.

This project combines OCR-based payment verification and Visual Product Search using feature matching. Currently, the Visual Product Search module has been fully implemented.

## Features Implemented

### Visual Product Search Module

The Visual Product Search allows sellers to quickly identify products from customer-uploaded images during live chat or streaming sessions. The implemented pipeline includes:

1. **Feature Extraction with SIFT**
   - Detects keypoints using **SIFT (Scale-Invariant Feature Transform)**.
   - Generates descriptors that are invariant to scale, rotation, and minor lighting changes.

2. **Feature Matching**
   - Uses **Brute Force** or **FLANN-based matcher** to compare query images against inventory images.
   - Computes similarity scores to find the most visually similar products.

3. **Top-K Retrieval and Visualization**
   - Ranks products based on similarity scores.
   - Displays top matches visually for quick verification by the seller.

  - Detects keypoints using SIFT (Scale-Invariant Feature Transform).
  - Generates descriptors that are invariant to scale, rotation, and minor lighting changes.
2. Feature Matching:
  * Uses Brute Force or FLANN-based matcher to compare query images against inventory images.
  * Computes similarity scores to find the most visually similar products.
3. Top-K Retrieval and Visualization:
- Ranks products based on similarity scores.
- Displays top matches visually for quick verification by the seller.
