# Import accuracy score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from src.visulization.visulize import metrics_score



# # Function to predict and evaluate
def evaluate_model(model, X_test,y_test):
    y_pred_train_svm = model.predict(X_test)
    scores = metrics_score(y_test, y_pred_train_svm)

    return scores

