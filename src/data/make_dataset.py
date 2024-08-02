import pandas as pd

def load_and_preprocess_data(data_path):
    
    # Import the data from 'credit.csv'
    df = pd.read_excel(data_path)
    df=df.drop(['EmployeeNumber','Over18','StandardHours'],axis=1)    #
    return df