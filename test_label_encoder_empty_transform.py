import numpy as np
from sklearn.preprocessing import LabelEncoder

def test_issue_reproduction():
    # Test that LabelEncoder.transform works consistently with empty lists
    # regardless of the data types used during fitting
    
    # Case 1: Fit with integers
    le_int = LabelEncoder()
    le_int.fit([1, 2, 3])
    result_int = le_int.transform([])
    
    # Case 2: Fit with strings  
    le_str = LabelEncoder()
    le_str.fit(['a', 'b', 'c'])
    result_str = le_str.transform([])
    
    # Both should return empty arrays of the same shape and type
    assert len(result_int) == 0
    assert len(result_str) == 0
    assert result_int.shape == result_str.shape
    
    # Test with numpy arrays as well
    result_int_np = le_int.transform(np.array([]))
    result_str_np = le_str.transform(np.array([]))
    
    assert len(result_int_np) == 0
    assert len(result_str_np) == 0