"""Tests for heatmap visualization (R source: heatmap.3.R)."""

import os

import numpy as np
import pytest

from copykat.heatmap import _normalize_side, heatmap3


class TestHeatmap3:
    def test_basic_call(self):
        rng = np.random.RandomState(42)
        data = rng.randn(50, 30)
        result = heatmap3(data, row_cluster=True, col_cluster=False, show=False)
        assert "row_order" in result
        assert len(result["row_order"]) == 50

    def test_no_clustering(self):
        data = np.random.RandomState(42).randn(20, 10)
        result = heatmap3(data, row_cluster=False, col_cluster=False, show=False)
        np.testing.assert_array_equal(result["row_order"], np.arange(20))

    def test_save(self, tmp_path):
        data = np.random.RandomState(42).randn(20, 10)
        path = str(tmp_path / "test.png")
        heatmap3(data, save_path=path, show=False)
        assert os.path.exists(path)

    def test_use_raster_keeps_pdf_small(self, tmp_path):
        rng = np.random.RandomState(0)
        data = rng.randn(60, 800)
        vec_path = str(tmp_path / "vec.pdf")
        ras_path = str(tmp_path / "ras.pdf")
        heatmap3(data, save_path=vec_path, show=False, use_raster=False)
        heatmap3(data, save_path=ras_path, show=False, use_raster=True)
        vec_size = os.path.getsize(vec_path)
        ras_size = os.path.getsize(ras_path)
        # At 48k cells the fixed overhead (dendrogram + annotations + legend)
        # caps the ratio at ~4-5x; copykat-scale (~3M cells) gives ~50x.
        assert ras_size * 3 < vec_size, (
            f"raster PDF ({ras_size} B) should be far smaller than "
            f"vector PDF ({vec_size} B)"
        )

    def test_dedup_duplicate_side_tracks(self):
        """R copykat convention: side colours doubled via cbind/rbind. Dedup it."""
        colours = np.array(["red", "blue", "red", "blue", "red"])
        doubled_col = np.column_stack([colours, colours])  # (5, 2) duplicated
        doubled_row = np.array([colours, colours])  # (2, 5) duplicated
        assert _normalize_side(doubled_col, 5, "col").shape == (5, 1)
        assert _normalize_side(doubled_row, 5, "row").shape == (5, 1)

    def test_dedup_keeps_distinct_tracks(self):
        a = np.array(["red", "blue", "red"])
        b = np.array(["green", "green", "yellow"])
        stacked = np.column_stack([a, b])  # (3, 2) distinct
        assert _normalize_side(stacked, 3, "col").shape == (3, 2)

    def test_show_routes_through_ipython_display(self, monkeypatch):
        """``show=True`` must surface a PNG to IPython so Jupyter renders
        inline. Without this hook the figure is silently dropped."""
        from IPython.display import Image
        import IPython.display

        captured = []
        monkeypatch.setattr(IPython.display, "display", captured.append)

        data = np.random.RandomState(42).randn(10, 8)
        heatmap3(data, row_cluster=True, col_cluster=False, show=True)

        assert len(captured) == 1
        assert isinstance(captured[0], Image)

    def test_figsize_controls_inline_pixel_size(self, monkeypatch):
        """``figsize`` must actually change the rendered inline PNG dimensions.
        Regression for the bug where R-pheatmap-faithful width/height plumbing
        meant figsize was silently ignored on the Jupyter inline path."""
        import struct
        from IPython.display import Image
        import IPython.display

        def png_dims(data: bytes) -> tuple[int, int]:
            # PNG IHDR layout: 8-byte signature, 4-byte length, 4-byte "IHDR",
            # then big-endian uint32 width and uint32 height.
            assert data[:8] == b"\x89PNG\r\n\x1a\n"
            w, h = struct.unpack(">II", data[16:24])
            return w, h

        captured = []
        monkeypatch.setattr(IPython.display, "display", captured.append)

        data = np.random.RandomState(0).randn(20, 30)
        heatmap3(data, show=True, figsize=(4.0, 2.0))
        heatmap3(data, show=True, figsize=(8.0, 4.0))

        small_w, small_h = png_dims(captured[0].data)
        big_w, big_h = png_dims(captured[1].data)

        # 8x4 in @ 150 dpi vs 4x2 in @ 150 dpi → factor-of-2 on each axis.
        assert big_w == 2 * small_w, (small_w, big_w)
        assert big_h == 2 * small_h, (small_h, big_h)
        # Sanity-check absolute dimensions (4 in × 150 dpi = 600 px).
        assert small_w == 600 and small_h == 300, (small_w, small_h)

    def test_ward_alias_matches_scipy_ward(self):
        """``link_method="ward"`` must produce SciPy's Ward (=R ``ward.D2``)
        row order — *not* pheatmap's bare-``"ward"`` semantics, which is
        legacy R ``ward.D``.  This matches the R copykat README/vignette and
        the pre-refactor matplotlib behaviour."""
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        data = np.random.RandomState(7).randn(40, 25)
        scipy_order = leaves_list(linkage(pdist(data, metric="euclidean"), method="ward"))

        alias = heatmap3(data, row_cluster=True, col_cluster=False,
                         link_method="ward", show=False)
        explicit = heatmap3(data, row_cluster=True, col_cluster=False,
                            link_method="ward.D2", show=False)

        np.testing.assert_array_equal(alias["row_order"], scipy_order)
        np.testing.assert_array_equal(alias["row_order"], explicit["row_order"])

    def test_dendrogram_keys_only_when_clustered(self):
        """Match the matplotlib-era contract: dendrogram keys are present only
        when clustering happened on that axis."""
        data = np.random.RandomState(42).randn(10, 8)

        r = heatmap3(data, row_cluster=False, col_cluster=False, show=False)
        assert "row_dendrogram" not in r
        assert "col_dendrogram" not in r

        r = heatmap3(data, row_cluster=True, col_cluster=False, show=False)
        assert "row_dendrogram" in r
        assert "col_dendrogram" not in r

        r = heatmap3(data, row_cluster=True, col_cluster=True, show=False)
        assert "row_dendrogram" in r
        assert "col_dendrogram" in r
