import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def data_retrieval():
    df_credit = pd.read_csv('/workspaces/functional-kan/phase_1/data/german_credit_data.csv')
    df_credit['Risk'] = df_credit['Risk'].map({'good': 1, 'bad': 0})

    df_credit['Monthly pay'] = (df_credit["Credit amount"] / df_credit["Duration"])

    df_credit.head()

    X = df_credit.drop(columns='Risk')
    # X = pd.get_dummies(X, dtype='int')
    y = df_credit['Risk']

    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Define the transformations
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the data
    X_transformed = pd.DataFrame(pipeline.fit_transform(X))

    df = pd.concat([X_transformed, y], axis=1)
    return df
