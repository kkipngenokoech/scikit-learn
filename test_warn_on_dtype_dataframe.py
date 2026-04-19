import pytest
import warnings
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array
from sklearn.exceptions import DataConversionWarning

def test_issue_reproduction():
    """Test that warn_on_dtype works with pandas DataFrame."""
    # Create a DataFrame with integer dtype
    df = pd.DataFrame([[1, 2], [3, 4]], dtype=np.int32)
    
    # This should trigger a DataConversionWarning when converting int32 to float64
    # but currently it doesn't warn for DataFrames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_array(df, dtype=np.float64, warn_on_dtype=True)
        
    # The test should fail because no warning is raised for DataFrames
    # but it should raise a DataConversionWarning
    assert len(w) > 0, "Expected DataConversionWarning for DataFrame dtype conversion"
    assert any(issubclass(warning.category, DataConversionWarning) for warning in w), \
        "Expected DataConversionWarning but got: {}".format([warning.category for warning in w])