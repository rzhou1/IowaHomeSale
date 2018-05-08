import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import make_scorer, mean_squared_error
#%matplotlib inline

pd.set_option('display.mpl_style', 'default')
pd.set_option('display.width', 3000)

na_values=['NA', 'N/A']
data_train = pd.read_csv('train.csv', na_values=na_values)
data_test = pd.read_csv('test.csv', na_values=na_values)
data_all = pd.concat([data_train, data_test])

#feature engineering
def Simply_YearBuilt(data):
    bins = (1800, 1900, 1930, 1950, 1965, 1980, 1990, 1995, 2000, 2005, 2010)
    group_names = ['00', '30', '50', '65', '80', '90', '95', '100', '105', '110']
    categories = pd.cut(data['YearBuilt'], bins, labels=group_names)
    data['YearBuilt'] = categories
    return data

def Simply_YearRemod(data):
    bins = (1940, 1960, 1970, 1980, 1990, 1995, 2000, 2005, 2010)
    group_names = ['60','70','80','90','95','100','105','110']
    categories = pd.cut(data['YearRemodAdd'], bins, labels=group_names)
    data['YearRemodAdd'] = categories
    return data

def Categ_feature(data):
    Categ_features = ['Neighborhood', 'MSZoning', 'Foundation', 'ExterQual', 'BsmtQual', 'SaleCondition', 'GarageCars']
    for feature in Categ_features:
        data[feature] = data[feature].fillna(0)
        data[feature] = pd.Categorical(data[feature]).codes
    return data

def SScaler(data):
    norm_features = ['LotArea', 'TotalBsmtSF', 'GrLivArea', '1stFlrSF', 'GarageArea', 'MasVnrArea', 'OpenPorchSF', 'EnclosedPorch']
    for feature in norm_features:
        data[feature] = data[feature].fillna(0)
    Scaler = StandardScaler()
    Scaler = Scaler.fit(data[norm_features])
    dt = Scaler.transform(data[norm_features])
    dt = pd.DataFrame(dt)

    for i, feature in enumerate(norm_features):
        data.loc[:, feature] = dt.iloc[:, i]
    return data

def encode_features(data):
    features = ['YearBuilt','YearRemodAdd']
    for feature in features:
        le = preprocessing.LabelEncoder().fit(data[feature])
        data[feature] = le.transform(data[feature])
    return data

def feature_transform(data):
    data = Simply_YearBuilt(data)
    data = Simply_YearRemod(data)
    data = Categ_feature(data)
    data = SScaler(data)
    data = encode_features(data)
    return data

data_train=feature_transform(data_train)
data_test=feature_transform(data_test)

x = data_train[['LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'Neighborhood', 'MSZoning', 'Foundation', 'ExterQual', 'BsmtQual', 'SaleCondition', 'KitchenAbvGr', 'HalfBath', 'MasVnrArea','OpenPorchSF', 'EnclosedPorch']]
y = np.log(data_train['SalePrice'])

data_test = data_test[['LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'Neighborhood', 'MSZoning', 'Foundation', 'ExterQual', 'BsmtQual', 'SaleCondition', 'KitchenAbvGr','HalfBath', 'MasVnrArea','OpenPorchSF', 'EnclosedPorch']]

test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 1)


#models

mse_dict = {}

#model: Linear
lin =Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LassoCV(alphas = np.logspace(-3,2,5), cv=3, fit_intercept=False))])
lin.fit(x_train, y_train)
y_hat_lin = lin.predict(x_test)
mse_dict['Linear'] = mean_squared_error(y_hat_lin, y_test)
print 'Linear model:', lin

#model: DecisionTree
dtr = DecisionTreeRegressor()
parameters = {'criterion': ['mse'], 'max_depth': [5, 8, 10,15,20], 'max_features': ['log2', 'sqrt', 'auto'], 'min_samples_leaf': [2,3,5,8]}
grid = GridSearchCV(dtr, parameters, scoring = make_scorer(mean_squared_error), n_jobs=-1, verbose=1)
grid.fit(x_train, y_train)
dtr = grid.best_estimator_
y_hat_dtr = dtr.predict(x_test)
mse_dict['DecisionTree'] = mean_squared_error(y_test, y_hat_dtr)
print 'Decision Tree:', dtr

#Ensemble: Random Forest
rfr = RandomForestRegressor(n_jobs=-1)

parameters = {'n_estimators':[5,20,50,100],
             'max_features':['log2','sqrt','auto'],
             'criterion':['mse'],
              'max_depth':[5,10,20,50],
             'min_samples_split':[2,3,5],
             'min_samples_leaf':[1,2,3,5,8]}
grid = GridSearchCV(rfr, parameters, scoring = make_scorer(mean_squared_error),n_jobs=-1,verbose=1)
grid = grid.fit(x_train, y_train)
rfr=grid.best_estimator_
rfr.fit(x_train, y_train)

#serializing model using either pickle or joblib
# filename = rfr_final.sav
# pickle.dump(rfr, open(filename, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))

y_hat_rfr = rfr.predict(x_test)
mse_dict['RandomForest'] = mean_squared_error(y_test, y_hat_rfr)
print 'Random Forest:', rfr

#Ensemble: XGBoost
xgbr = xgb.XGBRegressor()

params = {'max_depth': [3,5,8,10], 'learning_rate': [0.1, 0.2, 0.3],
          'n_estimators': [10, 20, 50, 100], 'objective': ['reg:linear'],
          'gamma': [i/10.0 for i in range(1,5)], 'reg_alpha': [i/10.0 for i in range(0,6)],
          'reg_lambda': [0, 0.2, 0.5, 0.8, 1.0], 'min_child_weight': [2,3,5]}

grid = GridSearchCV(xgbr, params, cv=3, n_jobs=-1, verbose=1)
grid.fit(x_train, y_train)
xgbr = grid.best_estimator_
y_hat_xgb = xgbr.predict(x_test)
mse_dict['XGBoost'] = mean_squared_error(y_hat_xgb, y_test)
print 'XGBoost:', xgbr

#Ensemble: LightGBM

#Results plot
t = np.arange(len(y_test))
plt.figure(figsize=(9,7))
results = pd.DataFrame(data={'Actual': y_test, 'Linear': y_hat_lin, 'DecisionTree': y_hat_dtr, 'RandomForest': y_hat_rfr, 'XGBoost': y_hat_xgb})
results = results.sort_values(by = 'Actual', axis  = 0)
plt.plot(t, results.Actual, 'g-', lw=2, label = 'True')
plt.plot(t, results.Linear, 'r-', lw=2, label = 'Linear')
plt.plot(t, results.DecisionTree, 'b', lw=2, label = 'DecisionTree')
plt.plot(t, results.RandomForest, 'c', lw=2, label = 'RandomForest')
plt.plot(t, results.XGBoost, 'm', lw=2, label = 'XGBoost')
plt.legend(loc='best')
plt.title('Ames home: actual vs. prediction', fontsize=18)
plt.xlabel('home no.')
plt.ylabel('price')
plt.grid()
plt.show()

print mse_dict