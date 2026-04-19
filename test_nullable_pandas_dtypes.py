import pandas as pd
import pytest
from sklearn.utils.multiclass import unique_labels

def test_issue_reproduction():
    # Test nullable pandas dtypes that should work like their non-nullable counterparts
    # but currently fail due to becoming object dtype when converted to numpy
    
    # This should work (non-nullable dtypes)
    y_int64 = pd.Series([1, 2, 3], dtype='int64')
    y_float64 = pd.Series([1.0, 2.0, 3.0], dtype='float64') 
    y_bool = pd.Series([True, False, True], dtype='bool')
    
    # These should also work but currently fail (nullable dtypes)
    y_nullable_int = pd.Series([1, 2, 3], dtype='Int64')
    y_nullable_float = pd.Series([1.0, 2.0, 3.0], dtype='Float64')
    y_nullable_bool = pd.Series([True, False, True], dtype='boolean')
    
    # Non-nullable dtypes work fine
    unique_labels(y_int64)
    unique_labels(y_float64)
    unique_labels(y_bool)
    
    # Nullable dtypes should work the same way but currently raise ValueError
    # This will fail with: ValueError: Mix type of y not allowed, got types {'binary', 'unknown'}
    unique_labels(y_nullable_int)
    unique_labels(y_nullable_float) 
    unique_labels(y_nullable_bool)