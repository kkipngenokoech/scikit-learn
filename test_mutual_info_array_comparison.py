import numpy as np
import warnings
from sklearn.feature_selection.mutual_info_ import _estimate_mi

def test_issue_reproduction():
    # Create simple test data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    # Test with discrete_features as boolean array (should trigger the bug)
    discrete_features = np.array([True, False])
    
    # This should raise a FutureWarning or ValueError due to array == 'auto' comparison
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            _estimate_mi(X, y, discrete_features=discrete_features)
            # If no warning/error, check if any warnings were raised
            if len(w) > 0:
                # Look for numpy comparison warnings
                for warning in w:
                    if "comparison" in str(warning.message).lower() or "elementwise" in str(warning.message).lower():
                        assert False, f"Array comparison warning raised: {warning.message}"
        except (ValueError, FutureWarning) as e:
            if "comparison" in str(e) or "truth value" in str(e):
                assert False, f"Array comparison error: {e}"
    
    # Also test with indices array
    discrete_features = np.array([0])  # indices of discrete features
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            _estimate_mi(X, y, discrete_features=discrete_features)
            if len(w) > 0:
                for warning in w:
                    if "comparison" in str(warning.message).lower() or "elementwise" in str(warning.message).lower():
                        assert False, f"Array comparison warning raised: {warning.message}"
        except (ValueError, FutureWarning) as e:
            if "comparison" in str(e) or "truth value" in str(e):
                assert False, f"Array comparison error: {e}"