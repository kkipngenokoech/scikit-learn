import numpy as np
from sklearn.metrics.cluster import mutual_info_score

def test_issue_reproduction():
    # Test that mutual_info_score should work with string labels
    # This reproduces the regression described in the issue
    x = np.random.choice(['a', 'b'], size=20)
    
    # This should work without raising a ValueError
    # but currently fails with "could not convert string to float: 'b'"
    result = mutual_info_score(x, x)
    
    # The result should be a valid float (mutual info of identical labels should be > 0)
    assert isinstance(result, float)
    assert result >= 0.0