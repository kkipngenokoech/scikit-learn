import pytest
import warnings
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.exceptions import DataConversionWarning

def test_issue_reproduction():
    """Test that warn_on_dtype works with pandas DataFrame."""
    # Create a DataFrame with int64 dtype
    df = pd.DataFrame([[1, 2], [3, 4]], dtype=np.int64)
    
    # This should raise a DataConversionWarning because we're converting
    # from int64 to float64, but currently it doesn't
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_array(df, dtype=np.float64, warn_on_dtype=True)
        
        # Check that a warning was issued
        assert len(w) > 0, "Expected DataConversionWarning but no warning was issued"
        assert any(issubclass(warning.category, DataConversionWarning) for warning in w), \
            f"Expected DataConversionWarning but got: {[warning.category for warning in w]}"
        
        # Verify the conversion actually happened
        assert result.dtype == np.float64