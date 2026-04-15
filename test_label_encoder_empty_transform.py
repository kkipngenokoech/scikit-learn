import numpy as np
from sklearn.preprocessing import LabelEncoder

def test_issue_reproduction():
    # Test case 1: LabelEncoder fitted with integers
    le_int = LabelEncoder()
    le_int.fit([1, 2, 3])
    result_int = le_int.transform([])
    
    # Test case 2: LabelEncoder fitted with strings  
    le_str = LabelEncoder()
    le_str.fit(['a', 'b', 'c'])
    result_str = le_str.transform([])
    
    # Both should return empty arrays with consistent behavior
    assert len(result_int) == 0
    assert len(result_str) == 0
    assert isinstance(result_int, np.ndarray)
    assert isinstance(result_str, np.ndarray)