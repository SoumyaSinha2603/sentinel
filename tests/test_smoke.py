"""Smoke tests so CI has something green to run from day one."""

import sentinel


def test_version():
    assert sentinel.__version__ == "0.1.0"


def test_config_paths_resolve():
    from sentinel.config import ROOT, RAW_DIR

    assert ROOT.exists()
    assert RAW_DIR.name == "raw"
