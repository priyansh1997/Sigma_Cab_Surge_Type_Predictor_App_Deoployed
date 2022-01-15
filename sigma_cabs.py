
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def dataset_prep(a):
    df=pd.read_csv('sigma_cabs.csv')
    df_n = df.drop(['Trip_ID','Cancellation_Last_1Month','Confidence_Life_Style_Index','Gender','Life_Style_Index','Var1','Var2'],axis=1)
    df_n.dropna(inplace=True)
    df_n.reset_index(inplace=True)
    df_n.drop('index',axis=1,inplace=True)
    df_n_X=df_n.drop('Surge_Pricing_Type',axis=1)
    df_n_y=pd.DataFrame(df_n['Surge_Pricing_Type'])
    ip=pd.DataFrame(a,index=['Trip_Distance',	'Type_of_Cab',	'Customer_Since_Months',	'Destination_Type',	'Customer_Rating',	'Var3']).T
    df_n_X=pd.concat([df_n_X, ip])
    df_n_X.reset_index(inplace=True)
    df_n_X.drop('index',axis=1,inplace=True)
    float_columns=[]
    cat_columns=[]
    int_columns=[]
    for i in df_n_X.columns:
        if df_n_X[i].dtypes=='float':
            float_columns.append(i)
        elif df_n_X[i].dtypes=='int64':
            int_columns.append(i)
        elif df_n_X[i].dtypes=='object':
            cat_columns.append(i)

    cat_features_df_n = df_n_X[cat_columns]
    float_features_df_n = df_n_X[float_columns]
    int_features_df_n = df_n_X[int_columns]
    df_n_cat_features_dummies_le = cat_features_df_n.apply(LabelEncoder().fit_transform)
    temp_1 = np.concatenate((df_n_cat_features_dummies_le,float_features_df_n),axis=1)
    train_transformed_features = np.concatenate((temp_1,int_features_df_n),axis=1)
    train_transformed_features = pd.DataFrame(data=train_transformed_features)
    X = train_transformed_features.values[:,:]
    y = np.ravel(np.array(df_n_y['Surge_Pricing_Type'].values))
    
    return (X,y)


def Loaded_model(temp):
    x,y=dataset_prep(temp)
    loaded_model = joblib.load('finalized_model.sav')
    return loaded_model.predict(x[-1].reshape(1,-1))[0]
