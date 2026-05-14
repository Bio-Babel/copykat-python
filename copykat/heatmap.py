"""Heatmap visualization (R source: heatmap.3.R), powered by pheatmap-python."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from pheatmap import colorRampPalette, pheatmap as _pheatmap
from scales import brewer_pal

__all__ = ["heatmap3"]


_RDBU_REVERSED = list(reversed(brewer_pal("div", "RdBu")(3)))


def _resolve_palette(cmap: Any, n: int) -> Sequence[str]:
    """Resolve a colormap spec to ``n`` hex colors."""
    if isinstance(cmap, (list, tuple)) and len(cmap) > 0:
        return colorRampPalette(list(cmap))(n)
    if isinstance(cmap, str):
        s = cmap.replace("-", "_")
        if s.lower() in {"rdbu_r", "rdbu_rev"}:
            return colorRampPalette(_RDBU_REVERSED)(n)
        if s.endswith("_r"):
            base = s[:-2]
            try:
                return colorRampPalette(list(reversed(brewer_pal("div", base)(3))))(n)
            except Exception:
                return colorRampPalette(list(reversed(brewer_pal("seq", base)(3))))(n)
        try:
            return colorRampPalette(brewer_pal("div", s)(3))(n)
        except Exception:
            return colorRampPalette(brewer_pal("seq", s)(3))(n)
    return colorRampPalette(_RDBU_REVERSED)(n)


def _normalize_side(arr: np.ndarray, n: int, axis: str) -> np.ndarray:
    """Coerce side-color array to shape (n, n_tracks), dropping duplicate tracks.

    The R copykat workflow doubles up a 1-D colour vector into a 2-column matrix
    (``cbind(CHR, CHR)``) so that base-R ``heatmap.3`` — which requires a matrix
    for ``ColSideColors`` / ``RowSideColors`` — renders a visually thicker band.
    pheatmap renders each track as a *labelled* annotation strip, so feeding it
    the doubled matrix produces two stacked identical bars.  We dedupe here so
    the visual matches base-R intent: one track per distinct colour series.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.shape[0] != n:
            raise ValueError(
                f"{axis}_side_colors length {arr.shape[0]} != number of {axis}s {n}"
            )
        return arr.reshape(-1, 1)
    if arr.shape[0] == n:
        out = arr
    elif arr.shape[1] == n:
        out = arr.T
    else:
        raise ValueError(
            f"{axis}_side_colors shape {arr.shape} does not match {axis} count {n}"
        )
    keep = [0]
    for j in range(1, out.shape[1]):
        if not all(np.array_equal(out[:, j], out[:, k]) for k in keep):
            keep.append(j)
    return out[:, keep]


def _build_annotation(
    arr: np.ndarray, index: Sequence[str], prefix: str
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    n_tracks = arr.shape[1]
    columns = [f"{prefix}{i + 1}" for i in range(n_tracks)]
    df = pd.DataFrame(arr.astype(str), columns=columns, index=list(index))
    colors = {c: {v: v for v in pd.unique(df[c])} for c in columns}
    return df, colors


def heatmap3(
    x: np.ndarray,
    *,
    row_cluster: bool = True,
    col_cluster: bool = False,
    dist_func: str = "euclidean",
    link_method: str = "ward",
    col_side_colors: np.ndarray | None = None,
    row_side_colors: np.ndarray | None = None,
    cmap: Any = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    breaks: np.ndarray | None = None,
    dendrogram_: str = "row",
    key: bool = True,
    figsize: tuple[float, float] = (10, 10),
    save_path: str | None = None,
    show: bool = False,
    use_raster: bool = True,
    interpolate: bool = False,
) -> dict[str, Any]:
    """Enhanced heatmap with optional dendrograms and categorical side tracks.

    Parameters
    ----------
    x
        Data matrix (rows x columns).
    row_cluster, col_cluster
        Toggle hierarchical clustering on each axis.
    dist_func
        Distance metric (R/pheatmap convention: ``"euclidean"``, ``"correlation"``,
        ``"manhattan"``, ...).
    link_method
        Linkage method (``"ward"``, ``"ward.D"``, ``"ward.D2"``, ``"complete"``, ...).
    col_side_colors, row_side_colors
        Color string arrays. Accepts ``(n,)``, ``(n, k)`` or ``(k, n)``;
        rendered as one or more annotation tracks.
    cmap
        Colormap spec. Accepts ``"RdBu_r"`` / ``"RdBu"`` / any RColorBrewer name
        (with optional ``_r`` suffix to reverse), or an explicit list of hex/named
        colors.
    vmin, vmax
        Used to build a default linear ``breaks`` if ``breaks`` is omitted.
    breaks
        Custom break points (length ``n_colors + 1``).
    dendrogram_
        Which dendrogram(s) to render: ``"row"``, ``"column"``, ``"both"``, ``"none"``.
    key
        Whether to draw the color legend.
    figsize
        ``(width, height)`` in inches.
    save_path
        If provided, save the figure to this path (PDF / PNG inferred from suffix).
    show
        If ``True``, render to the interactive grid device.
    use_raster
        If ``True`` (default), the heatmap body is embedded as a single raster
        image. This keeps PDF file size bounded for matrices with millions of
        cells (typical for copykat outputs with thousands of genomic bins).
        Dendrograms, annotations and legend remain vector.
    interpolate
        Bilinear interpolation when ``use_raster`` is ``True``; default
        nearest-neighbour to preserve cell-colour edges.

    Returns
    -------
    dict
        ``row_order``, ``col_order`` (0-based dendrogram leaf orderings),
        ``row_dendrogram``, ``col_dendrogram`` (SciPy linkage matrices or ``None``).
    """
    data = np.asarray(x)
    n_rows, n_cols = data.shape
    row_names = [f"r{i}" for i in range(n_rows)]
    col_names = [f"c{j}" for j in range(n_cols)]

    if breaks is not None:
        breaks_arr = np.asarray(breaks, dtype=float)
        n_colors = len(breaks_arr) - 1
    else:
        n_colors = 100
        if vmin is not None and vmax is not None:
            breaks_arr = np.linspace(vmin, vmax, n_colors + 1)
        else:
            extreme = float(np.nanmax(np.abs(data)))
            breaks_arr = np.linspace(-extreme, extreme, n_colors + 1)
    color = _resolve_palette(cmap, n_colors)

    th_row = None if dendrogram_ in ("row", "both", "r") else 0
    th_col = None if dendrogram_ in ("column", "both", "c") else 0

    ann_row: pd.DataFrame | None = None
    ann_col: pd.DataFrame | None = None
    ann_colors: dict[str, Any] = {}
    if row_side_colors is not None:
        rsc = _normalize_side(row_side_colors, n_rows, "row")
        ann_row, c = _build_annotation(rsc, row_names, "row_track_")
        ann_colors.update(c)
    if col_side_colors is not None:
        csc = _normalize_side(col_side_colors, n_cols, "col")
        ann_col, c = _build_annotation(csc, col_names, "col_track_")
        ann_colors.update(c)

    mat = pd.DataFrame(data, index=row_names, columns=col_names)

    ph = _pheatmap(
        mat,
        color=color,
        breaks=breaks_arr,
        cluster_rows=row_cluster,
        cluster_cols=col_cluster,
        clustering_distance_rows=dist_func,
        clustering_distance_cols=dist_func,
        clustering_method=link_method,
        treeheight_row=th_row,
        treeheight_col=th_col,
        legend=key,
        annotation_row=ann_row,
        annotation_col=ann_col,
        annotation_colors=ann_colors or None,
        annotation_names_row=False,
        annotation_names_col=False,
        annotation_legend=False,
        show_rownames=False,
        show_colnames=False,
        border_color=None,
        width=figsize[0],
        height=figsize[1],
        filename=save_path,
        silent=not show,
        use_raster=use_raster,
        interpolate=interpolate,
    )

    return {
        "row_order": np.asarray(ph.tree_row.order) if ph.tree_row is not None
                     else np.arange(n_rows),
        "col_order": np.asarray(ph.tree_col.order) if ph.tree_col is not None
                     else np.arange(n_cols),
        "row_dendrogram": ph.tree_row.linkage if ph.tree_row is not None else None,
        "col_dendrogram": ph.tree_col.linkage if ph.tree_col is not None else None,
    }
