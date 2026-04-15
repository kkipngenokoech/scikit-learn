import pytest
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array
from sklearn.exceptions import DataConversionWarning
import warnings

def test_issue_reproduction():
    """Test that warn_on_dtype works with pandas DataFrame."""
    # Create a DataFrame with integer dtype
    df = pd.DataFrame([[1, 2], [3, 4]], dtype=np.int32)
    
    # This should raise a DataConversionWarning because we're converting
    # from int32 to float64, but it doesn't on the buggy code
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_array(df, dtype=np.float64, warn_on_dtype=True)
        
        # Check that a DataConversionWarning was issued
        conversion_warnings = [warning for warning in w 
                             if issubclass(warning.category, DataConversionWarning)]
        assert len(conversion_warnings) > 0, "Expected DataConversionWarning but none was issued"
        
        # Verify the conversion actually happened
        assert result.dtype == np.float64
        assert np.array_equal(result, [[1.0, 2.0], [3.0, 4.0]])