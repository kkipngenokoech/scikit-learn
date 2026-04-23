import pandas as pd
import pytest
from sklearn.utils.multiclass import unique_labels

def test_issue_reproduction():
    # Test nullable pandas dtypes that should work like their non-nullable counterparts
    # but currently fail due to being converted to object dtype
    
    # Create pandas Series with nullable dtypes
    int_series = pd.Series([1, 2, 3], dtype="Int64")
    float_series = pd.Series([1.0, 2.0, 3.0], dtype="Float64") 
    bool_series = pd.Series([True, False, True], dtype="boolean")
    
    # These should work but currently raise ValueError due to 'unknown' type classification
    with pytest.raises(ValueError, match="Mix type of y not allowed, got types"):
        unique_labels(int_series)
    
    with pytest.raises(ValueError, match="Mix type of y not allowed, got types"):
        unique_labels(float_series)
        
    with pytest.raises(ValueError, match="Mix type of y not allowed, got types"):
        unique_labels(bool_series)