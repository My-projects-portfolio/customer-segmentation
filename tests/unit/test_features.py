import pandas as pd

from pipelines.features import pd as _pd  # dummy import to ensure module loads



def test_import_ok():
    assert True