from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", ["app.py", "bench.py", "check_gpu.py", "classify_images.py"])
def test_scripts_compile(script, tmp_path):
    py_compile.compile(str(ROOT / script), cfile=str(tmp_path / "out.pyc"), doraise=True)


def test_gui_builds_and_exposes_options():
    tk = pytest.importorskip("tkinter")
    import app

    try:
        window = app.ImageClassifierApp()
    except tk.TclError:
        pytest.skip("no display available")
    try:
        window.update()
        assert window.fast_decode_var.get() is False
        assert not hasattr(window, "engine_var")
    finally:
        window.destroy()
