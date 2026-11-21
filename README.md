# Multi-Modal Spatial Transcriptomics & Tissue Morphology Integration

A portable, command-line Python bioinformatics pipeline demonstrating **spatial multi-omics data integration**, **quantitative histology feature extraction**, and **AI-supported cross-modal phenotype prediction** built on the `scverse` ecosystem (`Scanpy`, `Squidpy`, `MuData`, `scikit-image`, and `scikit-learn`).

## Biological Motivation
Spatial multi-omics platforms (such as **DBiT-seq**, **10x Visium**, and **COMET** spatial proteomics) capture molecular readouts alongside high-resolution microscopy from the same tissue section. Integrating spatial gene expression with quantitative tissue morphology is critical for dissecting cellular heterogeneity and mapping tumor–immune microenvironments in human cancers such as melanoma.

## Pipeline Architecture
1. **Transcriptomic Preprocessing (`Scanpy`):** Quality control filtering, count normalization, log-transformation, highly variable gene (HVG) selection, PCA/UMAP dimensionality reduction, and Leiden spatial domain clustering.
2. **Quantitative Image Feature Extraction (`scikit-image` & `Squidpy`):** Extracts per-spot intensity statistics and Sobel edge-gradient features from paired tissue microscopy images to quantify cellular density and structural boundaries.
3. **Multi-Modal Data Integration (`MuData`):** Synchronizes spatial gene expression (`AnnData`) and quantitative morphology features into a unified `MuData` container.
4. **Spatial Microenvironment Statistics (`Squidpy`):** Maps user-defined marker genes, constructs spatial neighbor graphs to quantify **neighborhood enrichment** (Z-score co-localization vs. spatial exclusion), and identifies **spatially variable genes (SVGs)** via Moran's *I* autocorrelation.
5. **AI-Supported Cross-Modal Prediction (`scikit-learn`):** Trains a Random Forest classifier to predict molecular tissue domains directly from extracted histological morphology features.

## Case Study: Human Melanoma (10x Genomics Visium IF)
Validated on the public **10x Genomics Human Melanoma (IF Stained FFPE)** dataset (`--markers MLANA,PMEL,PTPRC,CD8A`):

* **Spatial Tissue Architecture:** Unsupervised Leiden clustering resolves **10 distinct histological niches**, clearly separating the stratified epidermis (clusters 1 and 6) from the deeper dermal tumor core (clusters 0, 3, and 4).
* **Tumor–Immune Microenvironment:** Canonical melanoma antigen **`MLANA`** densely populates the central tumor mass, whereas **`CD8A+`** cytotoxic T-cells and **`PTPRC+`** leukocytes remain largely restricted to the tumor periphery.
* **Neighborhood Enrichment (`Squidpy`):** Z-score spatial enrichment confirms strong intra-domain clustering, local intermixing between tumor sub-niches 3 and 4, and strong spatial exclusion between epidermal and deep tumor compartments.
* **Spatially Variable Genes (Moran's *I*):** Unbiased spatial autocorrelation identifies epidermal differentiation markers **`FLG`**, **`LOR`**, and **`FLG2`** sharply localized along the stratum granulosum/corneum boundary (cluster 6).

| Spatial Molecular Domains | Target Tumor & Immune Markers |
| :---: | :---: |
| ![Spatial Domains](figures/human_brain/01_spatial_domains.png) | ![Markers](figures/human_malonema/02_target_markers.png) |

| Spatial Neighborhood Enrichment | Top Spatially Variable Genes |
| :---: | :---: |
| ![Enrichment](figures/human_malonema/03_neighborhood_enrichment.png) | ![SVGs](figures/human_malonema/04_spatially_variable_genes.png) |

## Repository Structure
```text
spatial-multiomics-tissue-architecture/
├── README.md                 # Project overview, biological findings, and figures
├── requirements.txt          # Core scverse and machine learning dependencies
├── run_pipeline.sh           # Cross-platform CLI runner (auto-checks dependencies)
├── .gitignore                # Excludes local datasets, logs, and cache files
├── figures/                  # Output publication-grade spatial visualizations
└── src/
    └── pipeline.py           # Tissue-agnostic CLI spatial multi-omics pipeline
```

## Quickstart & CLI Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Mark6278/spatial-multiomics-tissue-architecture.git
cd spatial-multiomics-tissue-architecture
```

### 2. Prepare a 10x Visium Dataset
Download the **Human Melanoma, IF Stained (FFPE)** dataset from [10x Genomics Datasets](https://10xgenomics.com) (*filtered feature-barcode matrix `.h5`* and *spatial imaging `.tar.gz`*) and extract them into `data/melanoma/`:
```bash
mkdir -p data/melanoma
# Place your .h5 file inside data/melanoma/ and extract the spatial archive:
tar -xzf Visium_FFPE_Human_Melanoma_spatial.tar.gz -C data/melanoma/
```

### 3. Run the Pipeline (Any Tissue Dataset)
Execute via the Bash runner or directly with Python using custom CLI flags:
```bash
# Default run (Human Melanoma)
chmod +x run_pipeline.sh
./run_pipeline.sh --tissue-name "Human Melanoma"

# Example: Run on any other 10x Visium tissue (e.g., Breast Cancer)
python src/pipeline.py \
  --data-dir data/breast_cancer \
  --output-dir figures/breast_cancer \
  --tissue-name "Breast Cancer" \
  --markers "EPCAM,ERBB2,CD3D,CD68" \
  --resolution 0.6
```

## Future Extensions
* **Single-Cell Spatial Resolution:** Extend the `MuData` container to 10x Xenium and COMET multiplex immunofluorescence datasets using `Cellpose` nuclei segmentation masks.
* **Deep Multi-Modal Embeddings:** Integrate variational autoencoders (`scVI` / `TotalVI`) to jointly embed spatial transcriptomics and multiplexed protein layers.
