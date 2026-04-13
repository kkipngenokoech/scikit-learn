import numpy as np
from sklearn.preprocessing import LabelEncoder

def test_issue_reproduction():
    # Test case 1: LabelEncoder fitted with string data
    le_str = LabelEncoder()
    le_str.fit(['a', 'b', 'c'])
    
    # This should work but might fail
    result_str = le_str.transform([])
    assert len(result_str) == 0
    
    # Test case 2: LabelEncoder fitted with numeric data  
    le_num = LabelEncoder()
    le_num.fit([1, 2, 3])
    
    # This should work but might fail
    result_num = le_num.transform([])
    assert len(result_num) == 0
    
    # Both should return empty arrays of the same type
    assert isinstance(result_str, np.ndarray)
    assert isinstance(result_num, np.ndarray)