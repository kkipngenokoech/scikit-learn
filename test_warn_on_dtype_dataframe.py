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
    
    # Request float64 dtype with warn_on_dtype=True
    # This should trigger a DataConversionWarning but currently doesn't
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_array(df, dtype=np.float64, warn_on_dtype=True)
        
        # Check that a DataConversionWarning was raised
        dtype_warnings = [warning for warning in w 
                         if issubclass(warning.category, DataConversionWarning)]
        
        assert len(dtype_warnings) > 0, "Expected DataConversionWarning but none was raised"
        assert "Data with input dtype int32 was converted to float64" in str(dtype_warnings[0].message)