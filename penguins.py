import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
import catboost as cb

import shap
from streamlit_shap import st_shap


def prepare_data():

	# Import Data
	df = pd.read_csv('datasets/penguins.csv')
	st.write(df.head())
	df.dropna(axis=0, how='any', inplace=True)

	# Drop Columns
	COLS_TO_DROP = []
	df.drop(columns=COLS_TO_DROP, inplace=True)

	# Encode Binary Columns
	BINARY_COLS = ['Sex']  # leave a blank list if no binary cols
	if len(BINARY_COLS) > 0:
		class1 = df[BINARY_COLS[0]].value_counts().index[0]
		class2 = df[BINARY_COLS[0]].value_counts().index[1]
		df[BINARY_COLS[0]] = df[BINARY_COLS[0]].map({class1: 0, class2: 1})

	# Split X and y
	TARGET_COL = 'Species'
	X = df.drop(columns=TARGET_COL)
	y = df[TARGET_COL]

	# Encode Categorical Columns
	index = [x for x in X.dtypes.index]
	values = [x for x in X.dtypes.values]

	cols_to_ohe = []
	for i in range(len(index)):
		if values[i] == 'O' and index[i] not in BINARY_COLS:
			cols_to_ohe.append(index[i])

	for col in cols_to_ohe:
	    dummies = pd.get_dummies(X[col])
	    X = pd.concat((X, dummies), axis=1)
	    X = X.drop(columns=col)

	# Encode y Variable if Necessary
	if y.dtype == 'O':
	    if y.nunique() > 2:
	        encoding = 'categorical'
	    elif y.nunique() == 2:
	        encoding = 'binary'
	    else:
	        encoding = 'none'
	else:
	    encoding = 'none'

	if encoding == 'categorical':
		le = preprocessing.LabelEncoder()
		le.fit(pd.DataFrame(y))
		tf = le.transform(y)
		y = pd.DataFrame(tf)

	# Train/Test Split
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5, shuffle=True, random_state=123)


	cb_mdl = cb.CatBoostRegressor(depth=7, learning_rate=0.2,\
	                               random_state=123, verbose=False)

	cb_mdl = cb_mdl.fit(X_train, y_train)

	return X_train, X_test, cb_mdl



def run_penguins():

	X_train, X_test, cb_mdl = prepare_data()

	st.subheader('SHAP Beeswarm Plot')
	cb_explainer = shap.Explainer(cb_mdl)
	cb_shap = cb_explainer(X_test)
	st_shap(shap.plots.beeswarm(cb_shap, max_display=15))


	st.subheader('SHAP Feature Importance')
	st_shap(shap.plots.bar(cb_shap, max_display=15))
