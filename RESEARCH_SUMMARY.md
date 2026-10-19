# WasteRefine: Research Summary & Academic Brief

> **Document Type:** Pre-Publication Academic Brief & Project Abstract  
> **Status:** Under Peer Review (IEEE Conference)  
> **Affiliation:** BRAC University, Department of Computer Science and Engineering  
> **Authors:** Talha Islam Khan, Trisha Das, Md. Ahnaf Iqbal, Farhan Tawseef, Nirjhor Datta, Md. Golam Rabiul Alam  
> **Dataset Portal:** [WasteRefine on Roboflow Universe](https://universe.roboflow.com/talha-ljqb5/wasterefine)  
> **Date:** 2026  

---

## 📑 Visual Figures Index

All experimental charts and qualitative outputs in this repository follow standard figure identifiers:

| Figure Tag | File Path | Description |
|:---:|:---|:---|
| **Fig. 1** | [`results/fig1_model_architecture.png`](results/fig1_model_architecture.png) | End-to-end WasteRefine DINOv2-DPT architecture with PPM, SE attention, and BoundaryBranch |
| **Fig. 2** | [`results/fig2_dataset_samples.png`](results/fig2_dataset_samples.png) | Representative real-world annotated samples from the WasteRefine dataset |
| **Fig. 3** | [`results/fig3_benchmark_comparison.png`](results/fig3_benchmark_comparison.png) | Multi-dataset benchmark comparison against DeepLabV3+, COSNet, FANet, SegFormer |
| **Fig. 4** | [`results/fig4_per_class_results.png`](results/fig4_per_class_results.png) | Per-class IoU and Dice scores across all material categories on WasteRefine |
| **Fig. 5** | [`results/fig5_ablation_and_efficiency.png`](results/fig5_ablation_and_efficiency.png) | Decoder module ablation study (+2.95% gain) & parameter efficiency scatter plot |
| **Fig. 6** | [`results/fig6_qualitative_wasterefine.png`](results/fig6_qualitative_wasterefine.png) | Visual qualitative segmentation outputs on WasteRefine test split |
| **Fig. 7** | [`results/fig7_qualitative_zerowaste.png`](results/fig7_qualitative_zerowaste.png) | Visual qualitative segmentation outputs on ZeroWaste-F benchmark |
| **Fig. 8** | [`results/fig8_qualitative_spectralwaste.png`](results/fig8_qualitative_spectralwaste.png) | Visual qualitative segmentation outputs on SpectralWaste (RGB) benchmark |

---

## 📄 Abstract

The rapid acceleration of global municipal solid waste requires smart, data-driven frameworks for automated recyclable material identification and resource recovery. Conventional computer vision-based sorting systems deployed in Material Recovery Facilities (MRFs) frequently degrade in performance when confronted with:
1. Heavily cluttered and occluded scenes,
2. Extreme class imbalance where critical items (such as metals) are severely underrepresented, and
3. Deformed, crumpled, or translucent materials whose boundaries blend into neighboring garbage.

To address these challenges, we present **WasteRefine**, an object-aware semantic segmentation framework combining self-supervised Vision Foundation Models (DINOv2) with multi-scale feature decoding and boundary-guided feature refinement (see **Fig. 1**). 

Furthermore, we contribute the **WasteRefine Dataset**—the first comprehensive waste semantic segmentation benchmark collected from real-world roadside, landfill, and recycling environments in Bangladesh (visual samples displayed in **Fig. 2**). Comprising **2,213 precisely annotated images** and **10,860 annotated instances** across four essential recyclable categories (*Paper, Soft Plastic, Rigid Plastic, Metal*), WasteRefine establishes a regionally diverse, context-rich benchmark for automated waste sorting.

Extensive cross-benchmark evaluations across three challenging datasets (**WasteRefine**, **ZeroWaste-F**, and **SpectralWaste**) demonstrate state-of-the-art segmentation accuracy (see **Fig. 3**), exceeding notable published baselines while retaining high parameter efficiency (see **Fig. 5**).

---

## 🎯 Motivation & Research Questions

- **Why Foundation Models?**  
  Traditional CNNs trained from scratch or supervised backbones often overfit to domain-specific lighting and fail to capture wide-context scene dependencies. Self-supervised visual representations (SSL) preserve rich spatial priors that transfer exceptionally well to dense prediction tasks under cluttered backgrounds.
- **Why Boundary Enhancement?**  
  Crushed plastic bottles, torn paper, and compressed cans form overlapping piles where adjacent boundaries are ambiguous. Dedicated contour-level supervision compels the network to maintain high-frequency boundary separation between distinct material instances (see BoundaryBranch detail in **Fig. 1**).
- **Why a New Benchmark?**  
  Existing benchmarks either suffer from extreme imbalance (e.g., metal accounting for only ~1.4% of instances in ZeroWaste-F) or lack representation of developing countries where municipal waste composition differs dramatically from automated Western sorting lines (see dataset preview in **Fig. 2**).

---

## ⚙️ High-Level Methodology & System Workflow

The architecture illustrated in [**Fig. 1**](results/fig1_model_architecture.png) operates across three functional stages:

```
[ Input Conveyor Image ] 
           │
           ▼
┌──────────────────────────────────────────────┐
│       1. Foundation Feature Encoder          │
│   • Multi-layer token extraction (L3,6,9,11) │
│   • Global contextual representation         │
└──────────────────────┬───────────────────────┘
                       │ Multi-scale features (F₁ - F₄)
                       ▼
┌──────────────────────────────────────────────┐
│        2. Multi-Scale Decoder Path           │
│   • Global context pooling at deepest layer  │
│   • Channel-wise recalibration (SE blocks)   │
│   • Top-down progressive feature refinement  │
│   • Dedicated boundary-supervision branch    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│         3. Dual-Stream Output & Loss         │
│   • High-resolution semantic prediction      │
│   • Edge/contour delineation map             │
│   • Composite region + boundary optimization │
└──────────────────────────────────────────────┘
```

1. **Feature Extraction:** A pre-trained Vision Transformer processes images at native resolution, extracting intermediate representations across both early spatial layers ($F_1, F_2$) and deep semantic layers ($F_3, F_4$).
2. **Context & Attention Fusion:** Deeper features undergo multi-scale spatial pooling via a Pyramid Pooling Module (PPM) to construct global contextual priors, which are progressively integrated into higher-resolution representations with adaptive channel-attention (SE) mechanisms.
3. **Contour-Guided Separation:** A specialized edge prediction branch leverages shallow features ($F_1, F_2$) to enforce sharp class boundaries via dedicated structural supervision ($\mathcal{L}_{edge}$).
4. **Composite Training:** The model is optimized using a balanced loss combining region-level overlap metrics (Dice + Focal) and contour binary cross-entropy (BCE).

---

## 📊 Summary of Benchmark Results

All evaluations were conducted across three independent random seeds with mean and standard deviation reported below (visualized in [**Fig. 3**](results/fig3_benchmark_comparison.png) and [**Fig. 4**](results/fig4_per_class_results.png)):

### 1. Macro Benchmark Comparison (Refer to Fig. 3)
- **WasteRefine (Ours):**  
  - ViT-B achieved **96.64% mIoU** (and **96.80%** with Test-Time Augmentation).  
  - ViT-S achieved **96.26% mIoU** with only **25.16M parameters**, beating DeepLabV3+ (92.22%) and DINOv2-B Simple DPT (94.31%).
- **ZeroWaste-F (Industrial Benchmark):**  
  - ViT-B established a new state of the art at **61.94% mIoU**, significantly surpassing previous specialized architectures including COSNet (56.67%) and FANet (54.89%).  
  - ViT-S (25.16M) reached **59.75% mIoU**, outperforming models with up to 47% more parameters.
- **SpectralWaste (RGB Benchmark):**  
  - ViT-B achieved **70.73% Foreground mIoU**, outperforming the existing published benchmark record (COSNet: 69.96%).

### 2. Per-Class Consistency on WasteRefine (Refer to Fig. 4)
Every individual class exceeds **94% IoU** and **97% Dice Score**, demonstrating uniform generalization without class-dominance bias:
- **Metal:** 96.41% IoU (98.17% Dice)
- **Paper:** 96.95% IoU (98.45% Dice)
- **Rigid Plastic:** 95.29% IoU (97.59% Dice)
- **Soft Plastic:** 96.75% IoU (98.35% Dice)
- **Background:** 98.05% IoU (99.02% Dice)

### 3. Key Ablation Findings & Parameter Efficiency (Refer to Fig. 5)
Ablation experiments conducted on the ZeroWaste-F dataset validate the specific impact of each architectural component (see [**Fig. 5, Left**](results/fig5_ablation_and_efficiency.png)):
- Baseline DPT: **60.00% mIoU**
- Adding Context Pooling (PPM): **+0.46%**
- Adding Channel Attention (SE): **+0.79%**
- Adding Boundary Supervision (BoundaryBranch): **+0.85%**
- Adding Balanced Resampling (WRS): **+0.85%**
- **Total Cumulative Gain:** **+2.95% mIoU** over the baseline.

The parameter efficiency analysis (see [**Fig. 5, Right**](results/fig5_ablation_and_efficiency.png)) further highlights that ViT-S (25.16M) achieves superior accuracy over heavily parameterized CNN and Transformer alternatives.

---

## 🎨 Qualitative Demonstrations

Visual segmentation comparisons illustrating the input image, ground-truth annotations, GT overlays, and predictions from both ViT-B and ViT-S variants can be inspected in:
- [**Fig. 6** (`results/fig6_qualitative_wasterefine.png`)](results/fig6_qualitative_wasterefine.png): WasteRefine benchmark outputs.
- [**Fig. 7** (`results/fig7_qualitative_zerowaste.png`)](results/fig7_qualitative_zerowaste.png): ZeroWaste-F benchmark outputs under heavy clutter.
- [**Fig. 8** (`results/fig8_qualitative_spectralwaste.png`)](results/fig8_qualitative_spectralwaste.png): SpectralWaste (RGB) benchmark outputs on thin objects.

---

## 🔒 Code, Dataset, & Intellectual Property Notice

- **Intellectual Property Protection:**  
  Detailed component implementations, training schedules, parameter weights, and internal mathematical formulations are currently withheld to protect ongoing peer-review integrity and copyright.
- **Code Release:**  
  Cleaned training, evaluation, and inference pipelines, along with pre-trained checkpoints, will be released under an open-source license immediately following paper publication.
- **Dataset Access:**  
  Sample imagery and metadata are browsable on Roboflow Universe. The full dataset download will open upon final acceptance.

---

## 📖 Citation

```bibtex
@article{khan2026wasterefine,
  title   = {WasteRefine: Boundary-Aware Semantic Segmentation of Waste Materials Using a DINOv2 Backbone with Multi-Scale Feature Fusion Decoder},
  author  = {Talha Islam Khan and Trisha Das and Md. Ahnaf Iqbal and Farhan Tawseef and Nirjhor Datta and Md. Golam Rabiul Alam},
  journal = {Under Review},
  year    = {2026}
}
```

---

## 📬 Contact

For research inquiries or collaboration:  
- **Author:** [Md. Ahnaf Iqbal](mailto:md.ahnaf.iqbal@g.bracu.ac.bd)


