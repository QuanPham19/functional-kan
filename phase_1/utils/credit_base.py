import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config

def age_categorize(num):
    if num < 31:
        return '0'
    elif num < 55:
        return '1'
    # elif num < 60:
    #     return '2'
    else:
        return '2'

def housing_categorize(char):
    if char != 'own':
        return 'other'
    else:
        return char

def savings_categorize(char):
    if char in ['little', 'moderate']:
        return '0'
    else:
        return '1'

def credit_data_retrieval():
    df_credit = pd.read_csv('/workspaces/functional-kan/phase_1/data/german_credit_data.csv')
    df_credit['Risk'] = df_credit['Risk'].map({'good': 1, 'bad': 0})

    # Monthly pay
    df_credit['Monthly pay'] = (df_credit["Credit amount"] / df_credit["Duration"])

    # Age-amount interaction
    # df_credit['Age_Amount'] = df_credit['Age'] * df_credit['Credit amount']

    # Log credit amount
    # df_credit['Credit amount'] = np.log(df_credit['Credit amount'])

    # Age categorize
    df_credit['Age'] = df_credit['Age'].apply(age_categorize)

    # Housing categorize
    df_credit['Housing'] = df_credit['Housing'].apply(housing_categorize)

    # Savings categorize
    df_credit['Saving accounts'] = df_credit['Saving accounts'].apply(savings_categorize)

    # print(df_credit)


    X = df_credit.drop(columns='Risk')
    # X = pd.get_dummies(X, dtype='int')
    y = df_credit['Risk']

    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # set_config(transform_output='pandas')

    # Define the transformations
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the data
    X_transformed = pd.DataFrame(pipeline.fit_transform(X))

    df = pd.concat([X_transformed, y], axis=1)
    return df