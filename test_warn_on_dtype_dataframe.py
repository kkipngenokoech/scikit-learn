import pytest
import warnings
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.exceptions import DataConversionWarning

def test_issue_reproduction():
    """Test that warn_on_dtype works with pandas DataFrame."""
    # Create a DataFrame with object dtype that will be converted to float64
    df = pd.DataFrame({'a': ['1.0', '2.0', '3.0'], 'b': ['4.0', '5.0', '6.0']})
    assert df.dtypes['a'] == 'object'  # Ensure we start with object dtype
    
    # This should raise a DataConversionWarning but currently doesn't
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_array(df, warn_on_dtype=True)
        
        # Check that a warning was issued
        assert len(w) > 0, "Expected DataConversionWarning but no warning was issued"
        assert any(issubclass(warning.category, DataConversionWarning) for warning in w), \
            f"Expected DataConversionWarning but got: {[warning.category for warning in w]}"
        
        # Verify the conversion actually happened
        assert result.dtype == np.float64
        assert np.array_equal(result, np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))