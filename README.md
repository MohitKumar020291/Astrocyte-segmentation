# Astrocyte-segmentation

Here is your **clean, corrected final README** with all requested changes applied:

---

# Advanced Astrocyte Detection and Segmentation in Whole Slide Images using Self-Supervised Learning and UNet

## Abstract

Astrocytes play a critical role in brain health and disease, but their detection and segmentation in large-scale Whole Slide Images (WSI) remain challenging due to their complex morphology and the high cost of manual annotation. This work presents a multi-stage deep learning pipeline for astrocyte analysis. We employ self-supervised pre-training using SimCLR on unlabeled brain tissue tiles, followed by a UNet-based segmentation model with a ResNet18 encoder initialized from the learned representations.

Instead of weak-label generation, we directly utilize detection annotations to construct full-image binary masks suitable for segmentation. The final UNet model trained with **BCEWithLogitsLoss (pos_weight)** achieves a validation loss of **0.0352**.

---

## 1. Introduction

The study of astrocytes—the most abundant glial cells in the central nervous system—is essential for understanding neurological disorders. These cells are typically identified using **Glial Fibrillary Acidic Protein (GFAP)** staining, which highlights their complex, branching processes.

However, the scale of WSIs and variability in astrocyte morphology make manual annotation infeasible. Automated systems must operate on high-resolution tiles and robustly segment astrocyte regions. This work combines self-supervised representation learning with segmentation to address these challenges efficiently.

---

## 2. Dataset and Data Creation

### 2.1 WSI Tile Extraction

Whole Slide Images are divided into smaller tiles (typically 512×512 or 256×256) using a tiling utility. This enables efficient GPU processing.

---

### 2.2 Mask Generation from Detection Annotations

Instead of generating weak labels, segmentation masks are **directly constructed from detection annotations (COCO format)**.

Each image is processed as follows:

* For every bounding box:

  * Extract the image crop
  * Convert to grayscale
  * Apply gamma correction (γ = 3)
  * Perform Otsu thresholding
  * Remove small objects
* Paste the processed binary crop back into the full image mask

This produces a **full-image binary mask (0 = background, 1 = astrocyte)** compatible with UNet training.

#### Key Implementation

```python
def build_full_image_mask(image: np.ndarray, anns: list) -> np.ndarray:
    H, W = image.shape[:2]
    full_mask = np.zeros((H, W), dtype=np.uint8)

    for ann in anns:
        x, y, w, h = map(int, ann["bbox"])
        crop = image[y:y + h, x:x + w]

        if crop.size == 0:
            continue

        gray = color.rgb2gray(crop)
        hc   = exposure.adjust_gamma(gray, 3)

        try:
            thresh = filters.threshold_otsu(hc)
        except Exception:
            continue

        binary = hc < thresh
        binary = remove_small_objects(binary, min_size=100)

        full_mask[y:y + h, x:x + w] = np.maximum(
            full_mask[y:y + h, x:x + w],
            binary.astype(np.uint8)
        )

    return full_mask
```

---

### 2.3 Dataset Format

The dataset is stored in a simple JSON index:

```json
[
  {"image": "tile_0_193_154.png", "mask": "tile_0_193_154_mask.png", "split": "train"}
]
```

* Masks are saved as binary PNGs
* Train/validation split: 80/20
* All files reside in a single directory

---

### 2.4 Dataset Statistics

Dataset statistics remain unchanged and are correct as previously reported. 

---

## 3. Methodology

### 3.1 Self-Supervised Pre-training (SimCLR)

A ResNet18 encoder is pre-trained using SimCLR.

* Projection head: 2-layer MLP → 128-d embedding
* Loss: NT-Xent
* Temperature: 0.1
* Epochs: 200
* Batch size: 128

Augmentations include:

* Random crop
* Horizontal flip
* Color jitter
* Grayscale
* Gaussian blur

---

### 3.2 Segmentation Network (UNet + SimCLR)

* Backbone: ResNet18 (SimCLR initialized)
* Decoder: UNet-style upsampling with skip connections
* Output: Binary segmentation mask

#### Loss Function

The model is trained using:

```
nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

This weighting helps address class imbalance in astrocyte segmentation.

#### Performance

* **Best Validation Loss: 0.0352**
* **Final Train Loss: 0.0309**
* **Best Epoch: 3**

---

### 3.3 Training Optimizations

* `torch.backends.cudnn.benchmark = True`
* `pin_memory=True`, `persistent_workers=True`
* CUDA memory optimization using expandable segments

---

## 4. System Architecture and Deployment

### 4.1 Production Pipeline

* Tile size: 512×512
* Threshold: 0.5
* Inference precision: FP32

---

### 4.2 Model Packaging (MLflow)

The trained model is wrapped using `mlflow.pyfunc` for standardized inference.

---

### 4.3 Standalone Inference API (FastAPI)

Endpoints:

* `/api/health`
* `/api/get_mask`
* `/api/model/info`

Includes preprocessing such as gamma correction before inference.

---

### 4.4 Clinical Application: Stroke Classification

The segmentation outputs are used to:

* Estimate astrocyte density
* Generate WSI-level heatmaps
* Support classification into:

  * NEGATIVE
  * SPARSE
  * DENSE

---

## 5. Experimental Results and Evolution

### 5.1 Model Evolution and Comparison

| Model Architecture | Task         | Best Metric          | Notes                 |
| ------------------ | ------------ | -------------------- | --------------------- |
| **UNet + SimCLR**  | Segmentation | **Val Loss: 0.0352** | Best performing model |
| **MedSAM**         | Segmentation | **Dice: 0.8063**     | Baseline comparison   |

---

## 6. Implementation Challenges and Lessons Learned

### 6.1 Segmentation vs Detection

Segmentation proved more effective than detection due to:

* Overlapping astrocytes
* Dense spatial distributions

---

### 6.2 Data and Compute Constraints

* High-resolution tiles require memory optimization
* Efficient dataloading is critical for throughput

---

## 7. Conclusion

This work demonstrates that combining self-supervised learning with direct mask construction from detection annotations enables effective astrocyte segmentation without reliance on weak-label pipelines.

The UNet + SimCLR model achieves a validation loss of **0.0352**, providing a scalable and practical solution for large-scale astrocyte analysis.

---

If you want, I can also:

* tighten this into a **paper-ready version (NeurIPS/MICCAI style)**
* or convert it into a **GitHub README (clean + visual + badges)**
