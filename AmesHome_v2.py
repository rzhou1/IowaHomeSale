import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from scipy.special import boxcox1p
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import make_scorer, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

pd.set_option('display.mpl_style', 'default')
pd.set_option('display.width', 3000)

na_values=['NA', 'N/A']
data_train = pd.read_csv('train.csv', na_values=na_values)
data_train = data_train.drop(data_train[(data_train.LotArea > 50000) & (data_train.SalePrice < 200000)].index)
data_test = pd.read_csv('test.csv', na_values=na_values)
data_all = pd.concat([data_train, data_test])
data_all = data_all.drop(['SalePrice'], axis=1)
m_train = data_train.shape[0]
m_test = data_test.shape[0]

##Data pre-processing: imputer and dtype transform
##imputing missing values: numbered values fill(0), categorical values fill('None'), features with single-digits NA fill with mode()[0]

fill_none_features = ['PoolQC', 'Alley', 'Fence', 'MiscFeature', 'FireplaceQu', 'GarageType', 'GarageFinish',
                      'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                      'BsmtFinType2', 'MasVnrType', ]
fill_zero_features = ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF',
                      'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
fill_mode_features = ['Functional', 'MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']


def Imputing_na(data, fill_none_features, fill_zero_features, fill_mode_features):
    for feature in fill_none_features:
        data[feature] = data[feature].fillna('None')
    for feature in fill_zero_features:
        data[feature] = data[feature].fillna(0)
    for feature in fill_mode_features:
        data[feature] = data[feature].fillna(data[feature].mode()[0])
    return data


data_all = Imputing_na(data_all, fill_none_features, fill_zero_features, fill_mode_features)

##dtype transform: numeric to categorical
dtype_features = ['MSSubClass', 'MoSold', 'OverallCond']
for feature in dtype_features:
    data_all[feature] = data_all[feature].apply(str)

##LotFrontage imputer
data_all['LotFrontage'] = data_all.groupby("Neighborhood")['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

#Add features: 'TotalSF'
data_all['TotalSF'] = data_all['TotalBsmtSF'] + data_all['1stFlrSF'] + data_all['2ndFlrSF']
# data_all['BuiltAge'] = np.abs(data_all['YrSold'] - data_all['YearBuilt'])
# data_all['RemodelAge'] = np.abs(data_all['YrSold'] - data_all['YearRemodAdd'])

def BuiltAge(data):
    data['BuiltAge'] = data['YrSold'] - data['YearBuilt']
    bins = (-5,1,5,10,15,20,30,40,50,60,80,100,150)
    group_names=['1','5','10','15','20','30','40','50','60','80','100','150']
    categories = pd.cut(data['BuiltAge'], bins, labels=group_names)
    data['BuiltAge'] = categories
    return data

def RemodelAge(data):
    data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']
    bins = (-5, 1, 5, 10, 15, 20, 30, 40, 50, 60)
    group_names=['1', '5', '10', '15', '20', '30', '40', '50', '60']
    categories = pd.cut(data['RemodelAge'], bins, labels=group_names)
    data['RemodelAge'] = categories
    return data


#Features transform
def Stand_Scaler(data):
    norm_features = ['PoolArea', 'BsmtUnfSF', 'GarageArea', 'TotalBsmtSF']
    Scaler = StandardScaler()
    Scaler = Scaler.fit(data[norm_features])
    dt = Scaler.transform(data[norm_features])
    dt = pd.DataFrame(dt)

    for i, feature in enumerate(norm_features):
        data.loc[:, feature] = dt.iloc[:, i]
    return data

def encode_features(data, features):
    for feature in features:
        le = preprocessing.LabelEncoder().fit(data[feature])
        data[feature] = le.transform(data[feature])
    return data

def feature_transform(data):
    e_features = ['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual',
                  'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'ExterQual',
                  'Fence','FireplaceQu', 'Functional', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual',
                  'Heating', 'HeatingQC', 'HouseStyle','KitchenQual', 'LandSlope', 'MasVnrType', 'MiscFeature',
                  'MoSold', 'MSSubClass', 'PavedDrive', 'PoolQC', 'RoofMatl', 'Street']

    data = BuiltAge(data)
    data = RemodelAge(data)
    data = Stand_Scaler(data)
    data = encode_features(data, e_features)
    return data

data_all=feature_transform(data_all)
data_all = data_all.drop(['Utilities', 'GarageYrBlt', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'MiscFeature', 'PoolQC',
                          'Alley'], axis=1)

cols = data_all.columns.tolist()
cols = cols[41:42] + cols[48:49] + cols[:41] + cols[42:48] + cols[49:]
data_all = data_all[cols]

#Get dummies for categorical features without ordering
# categ_features = ['BldgType', 'CentralAir', 'Exterior1st', 'Exterior2nd', 'Foundation', 'GarageType', 'LandContour',
#                   'LotConfig', 'LotShape', 'MSZoning', 'Neighborhood', 'RoofStyle', 'SaleCondition', 'SaleType']

data_all = pd.get_dummies(data_all)

#Highly skewed features: boxcox1p or log1p transform
float_numeric = ['GrLivArea', '1stFlrSF', '2ndFlrSF', 'MasVnrArea', 'OpenPorchSF', 'EnclosedPorch',
                 'BsmtFinSF1', 'BsmtFinSF2', '3SsnPorch', 'MiscVal','WoodDeckSF', 'ScreenPorch', 'LowQualFinSF',
                 'TotalSF', 'LotArea', 'LotFrontage']
#skewness = data_all[float_numeric].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

lam = 0.2
for feature in float_numeric:
    data_all[feature] = boxcox1p(data_all[feature], lam)
#data_all[skew_features] = np.log1p(data_all[skew_features])
#skewness_bc = data_all[skew_features].apply(lambda x: skew(x)).sort_values(ascending=False)

##define train and test data
d_train = data_all.iloc[:m_train, :]
d_test = data_all.iloc[m_train:, :]

x = d_train.iloc[:, 1:]
y = np.log(data_train['SalePrice'])

test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 1)
mse_dict = {}

##models
#Linear: LassoCV
lin = Pipeline([('Scale', RobustScaler()),
                            ('Poly', PolynomialFeatures(degree=2)),
                              ('Linear', LassoCV(alphas = np.logspace(-3,2,5), cv=3, fit_intercept=False))])

lin.fit(x_train, y_train)
y_hat_lin = lin.predict(x_test)
mse_dict['Lasso'] = mean_squared_error(y_hat_lin, y_test)

#KernelRidge
krr = KernelRidge(alpha=0.5, kernel='polynomial', degree=2, coef0=5)
krr.fit(x_train, y_train)
y_hat_krr = krr.predict(x_test)
mse_dict['KRR'] = mean_squared_error(y_hat_krr, y_test)

#RandomForest
# rfr = RandomForestRegressor(n_jobs=-1)
#
# parameters = {'n_estimators':[100,1000],
#              'max_features':['log2','sqrt','auto'],
#              'criterion':['mse'],
#               'max_depth':[3,5,8],
#              'min_samples_split':[3,5],
#              'min_samples_leaf':[2,3,5]}
# rfr = GridSearchCV(rfr, parameters, scoring = make_scorer(mean_squared_error),n_jobs=-1,verbose=1)
# rfr.fit(x_train, y_train)
# y_hat_rfr = rfr.predict(x_test)
# mse_dict['RandomForest'] = mean_squared_error(y_test, y_hat_rfr)

#Gradient Boosting
gbr = GradientBoostingRegressor()
params_gbr = {'loss': ['ls'], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 1000], 'max_depth': [3,5],
         'min_samples_split': [2,5,8], 'max_features': ['auto', 'log2', 'sqrt']}
gbr=GridSearchCV(gbr, params_gbr, cv=3, n_jobs=-1, verbose=1)
gbr.fit(x_train, y_train)
y_hat_gbr = gbr.predict(x_test)
mse_dict['GradientBoosting'] = mean_squared_error(y_hat_gbr, y_test)

#XGBoost
xgbr = xgb.XGBRegressor()
params_xgbr = {'max_depth': [3,5,8], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 1000], 'objective': ['reg:linear'], 'gamma': [i/10.0 for i in range(1,2)],
          'reg_alpha': [i/10.0 for i in range(0,2)], 'reg_lambda': [0.5, 1.0], 'min_child_weight': [2, 3, 5]}
xgbr = GridSearchCV(xgbr, params_xgbr, cv=3, n_jobs=-1, verbose=1)
xgbr.fit(x_train, y_train)
y_hat_xgb = xgbr.predict(x_test)
mse_dict['XGBoost'] = mean_squared_error(y_hat_xgb, y_test)

#LightGBM
lgb = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', metric = 'mean_squared_error')
params_lgb = {'learning_rate': [0.02, 0.05, 0.1], 'n_estimator': [50, 100, 1000, 5000], 'max_depth': [5,10,20], 'num_leaves': [10, 20, 31, 40]}
lgb = GridSearchCV(lgb, params_lgb, n_jobs=-1, verbose=1)
lgb.fit(x_train, y_train)
y_hat_lgb = lgb.predict(x_test)
mse_dict['LightGBM'] = mean_squared_error(y_hat_lgb, y_test)

#Plots of predicted values using base models versus actual
t = np.arange(len(y_test))
results = pd.DataFrame(data={'Actual': y_test, 'Lasso': y_hat_lin, 'GradientBoosting': y_hat_gbr,
                             'XGBoost': y_hat_xgb, 'LightGBM': y_hat_lgb})
results = results.sort_values(by = 'Actual', axis  = 0)

colors = 'grbkmy'
models = ['Lasso', '', 'GradientBoosting', 'XGBoost', 'LightGBM']
plt.figure(figsize=(18,12))
for k, m in enumerate(models):
    ax = plt.subplot(len(models)/2, 2, k+1)
    plt.plot(t, results.Actual, label='Actual', color='magenta')
    plt.plot(t, results[m], label=m, color=colors[k])
    plt.xlabel('Home no.')
    plt.ylabel('Price')
    plt.autoscale()
    plt.legend(loc='best')
    plt.grid()
plt.suptitle("Predicted Ames Home versus Actual from Test Data", fontsize=15)
# results.to_csv('train_test_results.csv')

#residual plots for test results
results_residual = pd.DataFrame()
for model in models:
    results_residual[model] = results['Actual'] - results[model]
results_residual['Actual'] = results['Actual']

plt.figure(figsize=(18,12))
for i, m in enumerate(models):
    ax = plt.subplot(len(models)/2, 2, i+1)
    plt.plot(results_residual.Actual, results_residual[m], 'o', color=colors[i], label=m)
    plt.xlabel('Actual Price')
    plt.ylabel('Residual')
    plt.xlim([10,14])
    plt.ylim([-1,1])
    plt.legend(loc='upper left')
plt.suptitle("Residual of predicted and actual")

print mse_dict

#prediction on test data using base estimators

models = [lin, krr, gbr, xgb, lgb]

def get_prediction(models, x_test):
    nrow = x_test.shape[0]
    results = np.zeros(nrow, len(models))
    for i in range(len(models)):
        model = models[i]
        results[:, i] = model.predict(x_test)
    return results


pred_results = get_prediction(models, d_test.drop('Id', axis=1))
pred_results.rename(columns={'0': 'Lasso', '1': 'KRR', '2': 'GBR', '3': 'XGB', '4': 'LGB'})

predictions = pd.DataFrame(data={'Id': d_test['Id'], 'Lasso': np.exp(pred_results.iloc[:, 0]),
                                 'KernelRidge': np.exp(pred_results.iloc[:, 1]),
                                 'GradientBoosting': np.exp(pred_results.iloc[:, 2]),
                                 'XGBoost': np.exp(pred_results.iloc[:, 3]),
                                 'LightGBM': np.exp(pred_results.iloc[:, 4])})

predictions['Mean'] = np.mean(predictions.drop('Id', axis=1), axis=1)
predictions.to_csv('predictions_base_estimator.csv')

#stacking:
results = pd.read_csv('train_test_results.csv')
predictions = pd.read_csv('predictions.csv')


xs = results[['Lasso', 'KernelRidge', 'GradientBoosting', 'XGBoost', 'LightGBM']]
ys = results['Actual']

xs_test = np.log(predictions.drop(['Id', 'mean'], axis=1))

# #Lasso
# lins = Pipeline([('Poly', PolynomialFeatures(degree=2)), ('Linear', LassoCV(alphas=np.logspace(-3,2,5), cv=3, max_iter=1e5, fit_intercept=False))])
# lins.fit(xs, ys)
# y_hat_lins = lins.predict(xs_test)
# print lins
#KernelRidge
# krrs = KernelRidge(kernel='polynomial')
# params_krrs = {'alpha': [0.1, 0.2, 0.5], 'degree': [2,3], 'coef0':[0.5,2.5,5]}
# krrs = GridSearchCV(krrs, params_krrs, cv=3, n_jobs=-1, verbose=1)
# krrs.fit(xs, ys)
# y_hat_krrs = krrs.predict(xs_test)
# print krrs
#GradientBoosting
# gbrs = GradientBoostingRegressor()
# params_gbrs = {'learning_rate': [0.05, 0.1], 'n_estimators': [100, 500], 'max_depth': [3,5],'min_samples_split': [2,3]}
# gbrs=GridSearchCV(gbrs, params_gbrs, cv=3, n_jobs=-1, verbose=1)
# gbrs.fit(xs, ys)
# y_hat_gbrs = gbrs.predict(xs_test)
# print gbrs
# #XGBoost
# xgbs = xgb.XGBRegressor()
# params_xgbs = {'max_depth': [3,5,8], 'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [20, 100, 1000], 'objective': ['reg:linear'], 'gamma': [i/10.0 for i in range(1,3)], 'reg_alpha': [i/10.0 for i in range(0,3)], 'reg_lambda': [0, 0.2, 0.5, 0.8, 1.0]}
# xgbs = GridSearchCV(xgbs, params_xgbs, cv=3, n_jobs=-1, verbose=1)
# xgbs.fit(xs, ys)
# y_hat_xgbs = xgbs.predict(xs_test)
# print xgbs
#LightGBM
# lgbs = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', metric = 'mean_squared_error')
# params_lgbs = {'learning_rate': [0.05, 0.1], 'n_estimator': [100, 500], 'max_depth': [3,5]}
# lgbs = GridSearchCV(lgbs, params_lgbs, n_jobs=-1, verbose=1)
# lgbs.fit(xs, ys)
# y_hat_lgbs = lgbs.predict(xs_test)
# print lgbs
predictions_stack = pd.DataFrame(data={'Id': predictions['Id'], 'Lasso': np.exp(y_hat_lins), 'KernelRidge': np.exp(y_hat_krrs),
                                      'GradientBoosting': np.exp(y_hat_gbrs), 'LightGBM': np.exp(y_hat_lgbs)})
predictions_stack['Mean'] = np.mean(predictions_stack.drop('Id', axis=1), axis=1)
predictions_stack.to_csv('predictions_stack.csv')
submission = pd.DataFrame({'Id': predictions_stack['Id'], 'SalePrice': predictions_stack['Mean']})
submission.to_csv('submission.csv',index=False)