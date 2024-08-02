# Created by RayLi
# This code implementated SVM method to train HR_employee data

from src.data.make_dataset import load_and_preprocess_data
from src.models.train_models import cluster
#from src.visulization.visulize import metrics_score
from src.feature_engineering.build_features import create_dummy_vars
from src.models.train_models import SVM
from src.models.predict_model import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "src/data/raw/HR_Employee_Attrition.xlsx"
    data = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    X,y = create_dummy_vars(data)
    svmmodel, X_test,y_test = SVM(X,y)
    
    # Evaluate the model
    accuracy = evaluate_model(svmmodel, X_test, y_test)
    
