import os
import joblib
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score,train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,mean_squared_error
from sklearn.compose import ColumnTransformer

MODEL_FILE = 'model.pkl'
PIPELINE = 'pipeline.pkl'

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline = Pipeline ([
     ('imputer', SimpleImputer(strategy = 'median')),
     ('scaler ', StandardScaler()),
     ])

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])

    full_pipeline = ColumnTransformer([
        ('num',num_pipeline,num_attribs),
        ('cat',cat_pipeline,cat_attribs),
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop('income_cat',axis=1).to_csv('input.csv',index=False)
        housing = housing.loc[train_index].drop("income_cat",axis=1)

    

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value',axis=1)

    num_attribs = housing_features.drop('ocean_proximity',axis=1).columns.tolist()
    cat_attribs = ['ocean_proximity']    
    
    pipeline=build_pipeline(num_attribs,cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)

    model.fit(housing_prepared,housing_labels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE)

    print("Model is Trained!")
else:
    model = joblib.load(MODEL_FILE)
    pipeline= joblib.load(PIPELINE)

    print('Inference complete.Result saved to output.csv')  
    
