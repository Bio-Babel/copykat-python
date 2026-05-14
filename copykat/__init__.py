"""copykat — Python port of the R copykat package for single-cell CNA inference."""

__version__ = "1.1.0"
__r_commit__ = "b7a4763"

from .annotate import annotate_genes
from .baseline import BaselineResult, baseline_gmm, baseline_norm_cl, baseline_synthetic
from .bins import BinConversionResult, convert_to_bins
from .copykat import CopykatResult, copykat
from .heatmap import heatmap3
from .segmentation import SegmentationResult, cna_mcmc
from .smoothing import dlm_smooth
from ._data import load_example_data, load_gene_annotations, load_dna_bins, load_cycle_genes

__all__ = [
    "copykat",
    "CopykatResult",
    "annotate_genes",
    "baseline_gmm",
    "baseline_norm_cl",
    "baseline_synthetic",
    "BaselineResult",
    "cna_mcmc",
    "SegmentationResult",
    "convert_to_bins",
    "BinConversionResult",
    "heatmap3",
    "dlm_smooth",
    "load_example_data",
    "load_gene_annotations",
    "load_dna_bins",
    "load_cycle_genes",
]
