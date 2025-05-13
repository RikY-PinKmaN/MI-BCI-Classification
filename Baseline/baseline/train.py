from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from mne.decoding import CSP

def train_model(x_train, y_train, x_test, y_test, n_components=3):
    """
    Trains a CSP + LDA model and evaluates its performance.
    """
    csp = CSP(n_components=n_components, reg='oas', log=True, norm_trace=False)
    csp.fit(x_train, y_train)
    x_train_csp = csp.transform(x_train)
    x_test_csp = csp.transform(x_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train_csp, y_train)
    y_pred = lda.predict(x_test_csp)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
