import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import svm

def test_issue_reproduction():
    # Create a simple pipeline with two steps
    anova_filter = SelectKBest(f_regression, k=5)
    clf = svm.SVC(kernel='linear')
    pipe = Pipeline([('anova', anova_filter), ('svc', clf)])
    
    # This should work but currently fails because Pipeline doesn't implement __len__
    with pytest.raises(TypeError, match="object of type 'Pipeline' has no len"):
        len(pipe)
    
    # This should also fail because len(pipe) fails
    with pytest.raises(TypeError):
        pipe[:len(pipe)]