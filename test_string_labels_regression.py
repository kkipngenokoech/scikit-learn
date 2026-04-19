import pytest
import numpy as np
from sklearn.metrics.cluster import mutual_info_score

def test_issue_reproduction():
    """Test that mutual_info_score fails with string labels due to input validation regression."""
    # This should work but currently fails with ValueError: could not convert string to float: 'b'
    x = np.random.choice(['a', 'b'], size=20)
    
    # This call should succeed but will fail due to the regression in check_clusterings
    with pytest.raises(ValueError, match="could not convert string to float"):
        mutual_info_score(x, x)