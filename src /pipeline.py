import argparse
import glob
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import mudata as md
from skimage.color import rgb2gray
from skimage.filters import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_arguments():
    """Parses command-line arguments so the pipeline can run on any spatial dataset."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Spatial Transcriptomics & Tissue Morphology Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "human_melanoma"),
        help="Path to directory containing 10x .h5 file and 'spatial/' folder (default: data/melanoma)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "figures/human_malonema"),
        help="Directory to save output figures (default: figures/)",
    )
    parser.add_argument(
        "--markers",
        type=str,
        default="MLANA,PMEL,PTPRC,CD8A",
        help="Comma-separated list of target marker genes to plot (default: MLANA,PMEL,PTPRC,CD8A)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Leiden clustering resolution (default: 0.5)",
    )
    parser.add_argument(
        "--tissue-name",
        type=str,
        default="Melanoma",
        help="Label used in plot titles (e.g., Melanoma, Breast Cancer)",
    )
    return parser.parse_args()


def load_spatial_data(data_dir):
    """Automatically detects and loads any 10x Genomics Visium dataset."""
    print(f"[1/6] Loading 10x Genomics spatial dataset from '{data_dir}'...")

    all_h5 = glob.glob(os.path.join(data_dir, "*.h5"))
    filtered_h5 = [f for f in all_h5 if "raw" not in os.path.basename(f).lower()]
    h5_files = filtered_h5 if filtered_h5 else all_h5

    if not h5_files:
        found = os.listdir(data_dir) if os.path.exists(data_dir) else "Directory not found"
        raise FileNotFoundError(
            f"No '.h5' file found in '{data_dir}'. Contents: {found}"
        )

    if not os.path.isdir(os.path.join(data_dir, "spatial")):
        raise FileNotFoundError(
            f"Missing 'spatial/' folder in '{data_dir}'. Extract your spatial .tar.gz archive first."
        )

    count_filename = os.path.basename(h5_files[0])
    print(f"      Found count matrix: {count_filename}")

    adata = sc.read_visium(path=data_dir, count_file=count_filename)
    adata.var_names_make_unique()

    sc.pp.calculate_qc_metrics(adata, inplace=True)
    sc.pp.filter_cells(adata, min_counts=500)
    sc.pp.filter_genes(adata, min_cells=10)

    print(f"      Retained {adata.n_obs} tissue spots and {adata.n_vars} genes after QC.")
    return adata


def preprocess_transcriptomics(adata, resolution=0.5, tissue_name="Melanoma"):
    """Normalizes counts, selects highly variable genes, and clusters spatial domains."""
    print("[2/6] Preprocessing transcriptomic layer with Scanpy...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added="spatial_domain")

    sq.pl.spatial_scatter(
        adata,
        color="spatial_domain",
        title=f"{tissue_name} Spatial Transcriptomic Domains",
        save="01_spatial_domains.png",
    )
    return adata


def extract_image_features(adata):
    """Extracts tissue morphology and Sobel edge-gradient features using scikit-image & Squidpy."""
    print("[3/6] Extracting histological/IF image features with scikit-image...")
    library_id = list(adata.uns["spatial"].keys())[0]
    img_array = adata.uns["spatial"][library_id]["images"]["hires"]
    hires_scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]

    if img_array.ndim == 3 and img_array.shape[2] >= 3:
        gray_img = rgb2gray(img_array[:, :, :3])
    elif img_array.ndim == 3:
        gray_img = img_array[:, :, 0]
    else:
        gray_img = img_array

    edge_img = sobel(gray_img)[..., np.newaxis]

    img_container = sq.im.ImageContainer(
        img_array, layer="hires", library_id=library_id, scale=hires_scale
    )
    img_container.add_img(
        edge_img,
        layer="sobel_edges",
        library_id=library_id,
        scale=hires_scale,
        dims=("y", "x", "channels_edge"),
    )

    sq.im.calculate_image_features(
        adata,
        img_container,
        layer="hires",
        features=["summary"],
        key_added="img_summary",
        spot_scale=1.0,
    )
    sq.im.calculate_image_features(
        adata,
        img_container,
        layer="sobel_edges",
        features=["summary"],
        key_added="edge_summary",
        spot_scale=1.0,
    )

    img_features = pd.concat(
        [adata.obsm["img_summary"], adata.obsm["edge_summary"]], axis=1
    ).fillna(0)
    img_features.columns = [f"morph_{col}" for col in img_features.columns]
    print(f"      Extracted {img_features.shape[1]} quantitative image features per spot.")
    return img_features


def build_multimodal_object(adata, img_features):
    """Synchronizes transcriptomics and tissue morphology into a MuData multi-omics object."""
    print("[4/6] Integrating RNA and Image modalities into MuData container...")
    morph_adata = sc.AnnData(
        X=img_features.values,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=img_features.columns),
    )
    morph_adata.obsm["spatial"] = adata.obsm["spatial"].copy()

    mdata = md.MuData({"rna": adata, "morphology": morph_adata})
    mdata.update()
    print(f"      Created MuData object with modalities: {list(mdata.mod.keys())}")
    return mdata


def analyze_spatial_microenvironment(adata, candidate_markers):
    """Maps user-specified marker genes and computes spatial neighborhood enrichment."""
    print("[5/6] Analyzing spatial microenvironment with Squidpy...")

    available_markers = [g.strip() for g in candidate_markers if g.strip() in adata.raw.var_names]
    missing_markers = [g.strip() for g in candidate_markers if g.strip() not in adata.raw.var_names]

    if missing_markers:
        print(f"      Note: Markers not found in dataset and skipped: {missing_markers}")

    if available_markers:
        print(f"      Plotting target markers: {available_markers}")
        sq.pl.spatial_scatter(
            adata,
            color=available_markers,
            use_raw=True,
            cmap="magma",
            ncols=2,
            save="02_target_markers.png",
        )

    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
    sq.gr.nhood_enrichment(adata, cluster_key="spatial_domain")

    sq.pl.nhood_enrichment(
        adata,
        cluster_key="spatial_domain",
        title="Spatial Domain Neighborhood Enrichment (Z-score)",
        save="03_neighborhood_enrichment.png",
    )

    hvg_subset = adata.var_names[adata.var["highly_variable"]][:100]
    sq.gr.spatial_autocorr(adata, mode="moran", genes=hvg_subset)
    top_svgs = adata.uns["moranI"].head(3).index.tolist()
    print(f"      Top Spatially Variable Genes (Moran's I): {top_svgs}")

    sq.pl.spatial_scatter(
        adata,
        color=top_svgs,
        use_raw=True,
        cmap="viridis",
        title=[f"Top SVG: {g}" for g in top_svgs],
        save="04_spatially_variable_genes.png",
    )


def predict_domains_from_morphology(mdata):
    """Trains a Random Forest classifier using local + spatially averaged morphology features."""
    print("[6/6] Training Random Forest model with spatial neighborhood feature aggregation...")
    X_local = mdata["morphology"].X
    y = mdata["rna"].obs["spatial_domain"].astype(str)

    # Compute spatial neighborhood average using Squidpy's spatial connectivity matrix
    conn = mdata["rna"].obsp["spatial_connectivities"]
    row_sums = np.asarray(conn.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # Prevent division by zero for isolated spots
    X_neigh = (conn @ X_local) / row_sums[:, np.newaxis]

    # Concatenate local spot features + 1-hop spatial neighborhood context
    X_combined = np.hstack([X_local, X_neigh])

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"      Spatial-Context Morphology Classification Accuracy: {acc:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)
    sc.settings.figdir = args.output_dir
    sc.settings.set_figure_params(dpi=150, facecolor="white", frameon=False)

    marker_list = args.markers.split(",")

    adata = load_spatial_data(data_dir=args.data_dir)
    adata = preprocess_transcriptomics(
        adata, resolution=args.resolution, tissue_name=args.tissue_name
    )
    img_features = extract_image_features(adata)
    mdata = build_multimodal_object(adata, img_features)
    analyze_spatial_microenvironment(mdata["rna"], candidate_markers=marker_list)
    predict_domains_from_morphology(mdata)

    print(f"\nAll steps completed! Figures saved to '{args.output_dir}/'.")
