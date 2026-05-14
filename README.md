# copykat_py

Python port of the R [**copykat**](https://github.com/navinlabcode/copykat) package for single-cell copy number aberration (CNA) inference from scRNA-seq data.

## Installation

```bash
pip install -e ".[dev]"
```

The CNA heatmap is rendered through [`pheatmap-python`](https://github.com/Bio-Babel/pheatmap-python) (which transitively pulls `rgrid-python` + `gtable-python` + `pycairo`). `pycairo` needs a system Cairo library at build time — on a typical conda environment `conda install -c conda-forge cairo pkg-config` is sufficient. On Debian / Ubuntu, `apt install libcairo2-dev pkg-config`.

## Quick start

```python
import copykat

res = copykat.copykat(rawmat=..., sam_name="test")
copykat.heatmap3(
    mat,
    col_side_colors=chr_colors,
    row_side_colors=pred_colors,
    breaks=col_breaks,
    save_path="cna_heatmap.pdf",
    use_raster=True,  # default — embeds the matrix body as one raster image
)
```

`use_raster=True` (default in `copykat.heatmap3`) keeps PDF file sizes bounded for matrices with millions of cells. Dendrograms, side colour tracks and the legend remain vector. Pass `use_raster=False` for a fully vector output.

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```
