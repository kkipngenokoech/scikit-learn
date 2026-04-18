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
    # The error occurs when trying to get len(pipe) for the slice operation
    result = pipe[:len(pipe)]
    
    # Verify that the result is the same pipeline (all steps)
    assert len(result.steps) == 2
    assert result.steps[0][0] == 'anova'
    assert result.steps[1][0] == 'svc'