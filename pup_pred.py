# load packages and read in the data

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib


# a function to do the training and prediction
def train_pred(n_sims, X, Y, f_names, test_size, pred_data):
    RMSE = np.zeros(n_sims)
    f_imp = np.zeros([n_sims, np.shape(X)[1]])
    pred_arr = np.zeros([n_sims, np.shape(pred_data)[0]])
    print("Init pred_arr of shape: " + str(pred_arr.shape))

    for i in range(n_sims):
        # split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        # initialize XGBRegressor
        GB = xgb.XGBRegressor()

        # the parameter grid below was too much on the kaggle kernel
        param_grid = {"learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
                      "objective": ['reg:linear'],
                      "n_estimators": [300, 500, 1000, 2000]}
        # do GridSearch
        search_GB = GridSearchCV(GB, param_grid, cv=4, n_jobs=-1).fit(X_train, Y_train)
        # the best parameters should not be on the edges of the parameter grid
        print('   ', search_GB.best_params_)
        
		# train the best model
        xgb_pups = xgb.XGBRegressor(**search_GB.best_params_).fit(X_train, Y_train)

        # predict on the test set
        test_pred = xgb_pups.predict(X_test)

        # feature importance
        b = xgb_pups.booster()
		# http://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster
        f_imp[i, :] = list(b.get_fscore().values()) # Get feature importance of each feature

        # rmse of prediction
        RMSE[i] = np.sqrt(mean_squared_error(Y_test, test_pred))
        
        pred_arr[i, :] = xgb_pups.predict(pred_data)
        pred = np.mean(pred_arr, axis=0)
    '''
    # visualize the prediction of the last model
    plt.scatter(Y_test, test_pred, label = 'regression model')
    plt.plot(np.arange(np.max(Y_test)), np.arange(np.max(Y_test)), color='k', label='perfect prediction')
    plt.title('predictions of the last model')
    plt.legend(loc='best')
    plt.xlabel('true #pups')
    plt.ylabel('predicted #pups')
    plt.show()
	''' 
    return RMSE, f_imp, pred