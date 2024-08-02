import pandas as pd

# create dummy features
def create_dummy_vars(df):
    #creating list of dummy columns
    to_get_dummies_for = ['BusinessTravel', 'Department','Education', 'EducationField','EnvironmentSatisfaction', 'Gender',  'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus' ]

    #creating dummy variables
    df = pd.get_dummies(data = df, columns= to_get_dummies_for, drop_first= True)

    #mapping overtime and attrition
    dict_OverTime = {'Yes': 1, 'No':0}
    dict_attrition = {'Yes': 1, 'No': 0}


    df['OverTime'] = df.OverTime.map(dict_OverTime)
    df['Attrition'] = df.Attrition.map(dict_attrition)
    
    Y= df.Attrition
    X= df.drop(columns = ['Attrition'])

    return X,Y