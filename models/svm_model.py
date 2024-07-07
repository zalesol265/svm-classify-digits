from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm(x_train, y_train, kernel='linear', C=1):
    svm_model = SVC(kernel=kernel, C=C)
    svm_model.fit(x_train, y_train)
    return svm_model

def evaluate_model(model, x_test, y_test):
    accuracy = model.score(x_test, y_test)
    return accuracy

def hyperparameter_tuning(x_train, y_train):
    param_grid = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
    grid_search = GridSearchCV(SVC(), param_grid, cv=3)
    grid_search.fit(x_train, y_train)
    return grid_search.best_params_
