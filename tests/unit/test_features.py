import pandas as pd

from pipelines.features import pd as _pd  # dummy import to ensure module loads


def test_dummy_imports():
    # Sanity check the module can be imported
    assert True
