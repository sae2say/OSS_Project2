import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import numpy as np


def sort_dataset(dataset_df):
    
    sorted_dataset_df = dataset_df.sort_values(by='year')
    return sorted_dataset_df

def split_dataset(dataset_df):

    dataset_df['salary'] = dataset_df['salary'] * 0.001
    x_train = dataset_df.iloc[:1718].drop(columns=['salary'])
    x_test = dataset_df.iloc[1718:].drop(columns=['salary'])
    y_train, y_test = dataset_df.iloc[:1718]['salary'], dataset_df.iloc[1718:]['salary']

    return x_train, x_test, y_train, y_test

def extract_numerical_cols(dataset_df):

    N_columns = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']

    for d_column in dataset_df.columns:
        if d_column not in N_columns:
            del dataset_df[d_column]

    return dataset_df

def train_predict_decision_tree(X_train, Y_train, X_test):

	model = DecisionTreeRegressor()
	model.fit(X_train, Y_train)
	prediction = model.predict(X_test)

	return prediction

def train_predict_random_forest(X_train, Y_train, X_test):
	
	model = RandomForestRegressor()
	model.fit(X_train, Y_train)
	prediction = model.predict(X_test)

	return prediction

def train_predict_svm(X_train, Y_train, X_test):
	
	model_pipe = make_pipeline(
		StandardScaler(),
		SVR()
	)

	model_pipe.fit(X_train, Y_train)
	prediction = model_pipe.predict(X_test)

	return prediction

def calculate_RMSE(labels, predictions):
	cal_RMSE = np.sqrt(np.mean((predictions-labels)**2))
	return cal_RMSE

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))