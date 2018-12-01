import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler, MinMaxScaler
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

def loadCombineData():
    na_values = ['NA', 'N/A']
    data_train = pd.read_csv('train.csv', na_values=na_values)
    data_train = data_train.drop(data_train[(data_train.LotArea > 50000) & (data_train.SalePrice < 200000)].index)
    data_test = pd.read_csv('test.csv', na_values=na_values)
    data_all = pd.concat([data_train, data_test])
    data_all = data_all.drop(['SalePrice'], axis=1)
    m_train = data_train.shape[0]
    return data_train, data_all, data_test, m_train

class Preprocess():
    def __init__(self):
        pass

    def _imputer(self, data):
        fillNoneFeatures = ['PoolQC', 'Alley', 'Fence', 'MiscFeature', 'FireplaceQu', 'GarageType', 'GarageFinish',
                            'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                            'BsmtFinType2', 'MasVnrType']
        fillZeroFeatures = ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF',
                            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
        fillModeFeatures = ['Functional', 'MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd','SaleType']
        for feature in fillNoneFeatures:
            data[feature] = data[feature].fillna('None')
        for feature in fillZeroFeatures:
            data[feature] = data[feature].fillna(0)
        for feature in fillModeFeatures:
            data[feature] = data[feature].fillna(data[feature].mode()[0])
        data['LotFrontage'] = data.groupby("Neighborhood")['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
        return data

    def _dtype_transform(self, data):
        dtypeFeatures = ['MSSubClass', 'MoSold', 'OverallCond']
        for feature in dtypeFeatures:
            data_all[feature] = data_all[feature].apply(str)
        return data

    def _bin_built_age(self, data):
        data['BuiltAge'] = data['YrSold'] - data['YearBuilt']
        bins = (-5, 1, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150)
        group_names = ['1', '5', '10', '15', '20', '30', '40', '50', '60', '80', '100', '150']
        categories = pd.cut(data['BuiltAge'], bins, labels=group_names)
        data['BuiltAge'] = categories
        return data

    def _bin_remodel_age(self, data):
        data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']
        bins = (-5, 1, 5, 10, 15, 20, 30, 40, 50, 60)
        group_names=['1', '5', '10', '15', '20', '30', '40', '50', '60']
        categories = pd.cut(data['RemodelAge'], bins, labels=group_names)
        data['RemodelAge'] = categories
        return data

    def _add_features(self, data):
        data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
        data['Seasonality'] = data['MoSold'].astype(int)//4
        return data

    def preprocess(self, data):
        data = self._imputer(data)
        data = self._dtype_transform(data)
        data = self._bin_built_age(data)
        data = self._bin_remodel_age(data)
        data = self._add_features(data)
        return data


class FeaturesTransform(object):
    def __init__(self):
        pass

    def _boxcox_transform(self, data, lam):
        '''To reduce skewness of numeric features.'''
        transformFeatureIndex = (0, 1, 2, 3, 4, 5, 9, 12, 15, 19, 20, 21, 22, 23, 24, 27, 30, 36)
        numericFeatures = data.select_dtypes(exclude='object').columns.tolist()
        transformNumericFeatures = [item for i, item in enumerate(numericFeatures) if i in transformFeatureIndex]
        skewness = data[transformNumericFeatures].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame(data={'skewness': skewness})
        skewTransformNumericFeatures = skewness[abs(skewness.skewness) > 0.5].index
        for feature in skewTransformNumericFeatures:
            data[feature] = boxcox1p(data[feature], lam)
        return data

    def _standard_scaler(self, data):
        normFeatures = ['BsmtUnfSF', 'GarageArea', 'PoolArea',
                        'TotalBsmtSF']  # better normalcy compared to boxcox1p / log1p
        Scaler = StandardScaler()
        Scaler = Scaler.fit(data[normFeatures])
        dt = Scaler.transform(data[normFeatures])
        dt = pd.DataFrame(dt)

        for i, feature in enumerate(normFeatures):
            data.loc[:, feature] = dt.iloc[:, i]
        return data

    def _encode_ordinal_features(self, data):
        '''Features with ordering.'''
        nonEncodingFeaturesIndex = (1, 7, 13, 14, 17, 22, 27, 29, 30, 32, 36, 41, 42, 43, 45)
        nonObjectOrdinalFeatures = ['BuiltAge', 'RemodelAge', 'GarageCars']
        objectOrdinalFeatures = data.select_dtypes(include='object').columns.tolist()
        ordinalFeatures = [item for i, item in enumerate(objectOrdinalFeatures) if i not in nonEncodingFeaturesIndex]
        ordinalFeatures.extend(nonObjectOrdinalFeatures)
        for feature in ordinalFeatures:
            le = preprocessing.LabelEncoder().fit(data[feature])
            data[feature] = le.transform(data[feature])
        return data

    def _drop_features(self, data):
        data = data.drop(
            ['Utilities', 'GarageYrBlt', 'YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd', 'MiscFeature', 'PoolQC',
             'Alley'], axis=1)
        return data

    def _encode_categorical_features(self, data):
        '''Features without apparent ordering.'''
        categoricalFeatures = ['BldgType', 'CentralAir', 'Exterior1st', 'Exterior2nd', 'Foundation', 'GarageType',
                               'LandContour',
                               'LotConfig', 'LotShape', 'MSZoning', 'Neighborhood', 'RoofStyle', 'SaleCondition',
                               'SaleType', 'Seasonality']
        data = pd.get_dummies(data=data, columns=categoricalFeatures)
        return data

    def transform(self, data):
        data = self._boxcox_transform(data, lam=0.2)
        data = self._standard_scaler(data)
        data = self._encode_ordinal_features(data)
        data = self._drop_features(data)
        data = self._encode_categorical_features(data)
        return data


# Split all the data into train, validation and test(unknown 'SalePrice'). Here we use train and validation for pre-tuning
# base models by GridSearchCV (not shown in "main" part).
def trainValTestSplit(data, data_train, test_size, random_state):
    '''part1. train and test(validation) data split.'''
    cols = data.columns.tolist()
    cols = cols[35:36] + cols[:35] + cols[36:]  # move 'ID' to the 1st column
    data = data[cols]
    x = data.iloc[:m_train, 1:]
    y = np.log1p(data_train['SalePrice'])
    x_test = data.iloc[m_train:, 1:]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_state)

    '''part2. drop features with zero standard deviations after split.'''
    zeroStdFeatures = []
    datasets = [x_train, x_val, x_test]
    dTemp = []
    for d in datasets:
        Describe = d.describe()
        dZeroStdFeatures = Describe.loc[:, Describe.loc['std', :] == 0].columns.tolist()
        zeroStdFeatures.extend(dZeroStdFeatures)
    for ds in datasets:
        ds = ds.drop(zeroStdFeatures, axis=1)
        dTemp.append(ds)

    x_train, x_val, x_test = dTemp

    return x_train, x_val, y_train, y_val, x_test


# Split the whole data into train (known 'SalePrice') and test (unknown 'SalePrice'), not validation. Here we run the base
# models by KFold that automatically place one fold for validation.
def trainTestSplit(data, data_train, random_state):
    '''part1. train and test data split.'''
    cols = data.columns.tolist()
    cols = cols[35:36] + cols[:35] + cols[36:]  # move 'ID' to the 1st column
    data = data[cols]
    x_train = data.iloc[:m_train, 1:]
    y_train = np.log1p(data_train['SalePrice'])
    x_test = data.iloc[m_train:, 1:]

    '''part2. drop features with zero standard deviations after split.'''
    zeroStdFeatures = []
    datasets = [x_train, x_test]
    dTemp = []
    for d in datasets:
        Describe = d.describe()
        dZeroStdFeatures = Describe.loc[:, Describe.loc['std', :] == 0].columns.tolist()
        zeroStdFeatures.extend(dZeroStdFeatures)
    for ds in datasets:
        ds = ds.drop(zeroStdFeatures, axis=1)
        dTemp.append(ds)

    x_train, x_test = dTemp

    return x_train, y_train, x_test


# Robust scaling and polynomial transform for linear (and generalized linear) models
class Pipeline(object):
    def __init__(self, x_train, x_test, degree):
        self.x_all = pd.concat([x_train, x_test])
        self.degree = degree

    def _robustScaler(self):
        transformer = RobustScaler().fit(self.x_all)
        self.x_all = transformer.transform(self.x_all)
        self.x_all = pd.DataFrame(data=self.x_all)

        return self.x_all

    def _polynomialFeatures(self):
        poly = PolynomialFeatures(degree=self.degree)
        self.x_all = poly.fit_transform(self.x_all)

        return self.x_all

    def transform(self):
        self.x_all = self._robustScaler()
        self.x_all = self._polynomialFeatures()
        self.x_all = pd.DataFrame(data=self.x_all)
        x_all_copy = self.x_all.copy()
        x_train_transform = x_all_copy.iloc[:m_train, :]
        x_test_transform = x_all_copy.iloc[m_train:, :]

        return x_train_transform, x_test_transform

#Define base models' training and predicting.
class BaseModels(object):
    def __init__(self, model, seed, params=None):
        #         params['random_state'] = seed
        self.model = model(**params)

    def train(self, x_train, y_train):
        return self.model.fit(x_train, y_train)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def kFold(self, x_train, y_train, x_test):
        n_folds = 5
        kf = KFold(x_train.shape[0], n_folds=n_folds, random_state=0)
        oof_train = np.zeros((x_train.shape[0],))
        oof_test_kf = np.zeros((n_folds, x_test.shape[0]))

        for i, (train_index, test_index) in enumerate(kf):
            x_train_kf = x_train.iloc[train_index]
            y_train_kf = y_train.iloc[train_index]
            x_test_kf = x_train.iloc[test_index]

            self.model.fit(x_train_kf, y_train_kf)

            oof_train[test_index] = self.model.predict(x_test_kf)
            oof_test_kf[i, :] = self.model.predict(x_test)

        oof_test = oof_test_kf.mean(axis=0)
        return oof_train, oof_test


def getBasePred(baseModels, params, x_train, y_train, x_test):
    '''Get prediction results from all base models.'''
    oof_train_hat = []
    oof_test_hat = []
    columns = {}
    baseMSE = {}
    for model, param in zip(baseModels, params):
        oof_train, oof_test = BaseModels(model, seed=0, params=param).kFold(x_train, y_train, x_test)
        oof_train_hat.append(oof_train)
        oof_test_hat.append(oof_test)
        baseMSE[model.__name__] = mean_squared_error(y_train, oof_train)

    for i in range(len(baseModels)):
        columns[i] = baseModels[i].__name__

    oof_train_hat, oof_test_hat = pd.DataFrame(data=oof_train_hat).T, pd.DataFrame(data=oof_test_hat).T
    oof_train_hat, oof_test_hat = oof_train_hat.rename(columns=columns), oof_test_hat.rename(columns=columns)

    return oof_train_hat, oof_test_hat, baseMSE


def stackingModel(models, x_train, y_train, x_test, n_folds):
    '''Model on prediction data by base models.'''
    stackPredTrain = []
    stackPredTest = []
    stackMSE = []

    for model in models:
        stackTrainOOF = np.zeros((x_train.shape[0],))
        stackTestOOF = np.zeros((n_folds, x_test.shape[0]))
        mseOOF = np.zeros((n_folds, 1))
        kf = KFold(x_train.shape[0], n_folds=n_folds, random_state=0)
        for i, (train_index, test_index) in enumerate(kf):
            x_train_oof = x_train.iloc[train_index]
            y_train_oof = y_train.iloc[train_index]
            x_test_oof = x_train.iloc[test_index]
            model.fit(x_train_oof, y_train_oof)
            stackTrainOOF[test_index] = model.predict(x_test_oof)
            mseOOF[i, :] = mean_squared_error(y_train.iloc[test_index], stackTrainOOF[test_index])
            stackTestOOF[i, :] = model.predict(x_test)
        stackTestOOF = stackTestOOF.mean(axis=0)
        stackPredTrain.append(stackTrainOOF)
        stackPredTest.append(stackTestOOF)
        mseOOF = mseOOF.mean(axis=0)
        stackMSE.append(mseOOF)

    stackPredTrain = pd.DataFrame(data=stackPredTrain).T
    stackPredTest = pd.DataFrame(data=stackPredTest).T
    stackMSE = pd.DataFrame(data=stackMSE)

    return stackPredTrain, stackPredTest, stackMSE


#Base prediction models: the hyperparameters for these models have been pre-tuned by GridSearchCV.

params_gbr = {'loss': 'ls', 'learning_rate': 0.03, 'n_estimators': 1000, 'max_depth': 3,'min_samples_split': 3,
              'max_features': 'sqrt'}
params_xgbr = {'max_depth': 5, 'learning_rate': 0.03, 'n_estimators': 500, 'objective': 'reg:linear',
               'min_child_weight': 5, 'gamma': 0.1, 'reg_lambda': 0.1}
params_Lasso = {'alpha': 0.01, 'max_iter':50, 'fit_intercept': False}
params_RF = {'n_estimators':100, 'max_features':'auto', 'max_depth':8, 'min_samples_split': 5, 'min_samples_leaf':2}
params_lgb = {'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 5, 'num_leaves': 10, 'boosting_type': 'gbdt',
              'objective': 'regression'}
params_Ridge = {'alpha': 0.01, 'max_iter': 100, 'fit_intercept': False}
params_KRR = {'alpha': 100, 'kernel': 'polynomial', 'degree': 2, 'coef0': 1000}
params_DT = {'max_depth': 8, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_features': 'auto'}
params_SVR = {'kernel': 'rbf', 'degree': 2, 'gamma': 'auto', 'coef0': 0.1, 'C': 1, 'max_iter': 1000}
params_ada = {'n_estimators': 1000, 'learning_rate': 0.3, 'loss': 'square'}

#Stacking (level 1) prediction models: similarly, the hyperparameters for these models have been pre-tuned by GridSearchCV.
xgbr = xgb.XGBRegressor(max_depth=8, learning_rate=0.1, n_estimators=100, gamma=0.1, objective='reg:linear',
                        booster='gbtree', reg_lambda=1.0, min_child_weight=2)
Lasso = Lasso(alpha=0.001, max_iter=50, fit_intercept=False)
Ridge = Ridge(alpha=0.3, max_iter=50, fit_intercept=False)
krr = KernelRidge(alpha=1, kernel='polynomial', degree=2, coef0=5)
gbr = GradientBoostingRegressor(learning_rate=0.01, criterion='friedman_mse', loss='ls', max_depth=3, n_estimators=500,
                                max_features='sqrt', min_samples_split=8)
lgbm = lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.1, max_depth=3, n_estimators=50, min_leaves=10,
                         lambda_l1=0.1, lambda_l2=1)
rf = RandomForestRegressor(n_estimators=50, criterion='mse', max_depth=8, max_features='log2', min_samples_split=8,
                           min_samples_leaf=2)
#svm = SVR(C=1, coef0=0.1, degree=2, gamma='auto', kernel='rbf', max_iter=1000)
ada = AdaBoostRegressor(learning_rate=0.01, n_estimators=100, loss='square')
dt = DecisionTreeRegressor(max_depth=5, max_features='sqrt', min_samples_leaf=2, min_samples_split=5)

if __name__ == '__main__':
    '''Data loading, preprocessing, feature engineering and train/test spliting.'''

    data_train, data_all, data_test, m_train = loadCombineData()  # load data
    data_all = Preprocess().preprocess(data_all)  # data preprocess
    data_all = FeaturesTransform().transform(data_all)  # feature transform
    x_train, y_train, x_test = trainTestSplit(data_all, data_train, random_state=0)  # train and test data split
    x_train_poly2, x_test_poly2 = Pipeline(x_train, x_test, 2).transform()  #Polynomialfeature transform for Linear models

    '''Base models prediction (KFold cross-validation).'''

    baseParams = [params_gbr, params_xgbr, params_RF, params_lgb, params_Ridge, params_KRR, params_DT, params_SVR, params_ada]
    baseModels = [GradientBoostingRegressor, xgb.XGBRegressor, RandomForestRegressor, lgb.LGBMRegressor, Ridge,
                  KernelRidge, DecisionTreeRegressor, SVR, AdaBoostRegressor]
    baseParamsLin = [params_Lasso]
    baseModelsLin = [Lasso]
    oof_train_hat, oof_test_hat, baseMSE = getBasePred(baseModels, baseParams, x_train, y_train, x_test)
    oof_train_hat_lasso, oof_test_hat_lasso, baseMSELasso = getBasePred(baseModels_lin, baseParams_lin, x_train_poly2,
                                                                     y_train, x_test_poly2)

    base_train, base_test, baseMSE = pd.DataFrame(data=oof_train_hat), pd.DataFrame(data=oof_test_hat), pd.DataFrame(data=baseMSE)
    base_train['Lasso'], base_test['Lasso'], baseMSE['Lasso'] = oof_train_hat_lasso.Lasso, oof_test_hat_lasso.Lasso, baseMSELasso
    print baseMSE

    '''Stacking model prediction.'''
    '''Stack: level-1, stack predictions from base models.'''

    stackingModels = [xgbr, Lasso, Ridge, krr, gbr, lgbm, rf, ada, dt]
    stackPredTrain, stackPredTest, stackMSE = stackingModel(stackingModels, base_train, y_train, base_test, 5)
    columns = {0: 'XGBoost', 1: 'Lasso', 2: 'Ridge', 3: 'KRR', 4: 'GradientBoosting', 5: 'LightGBM', 6: 'RandomForest',
               7: 'AdaBoost', 8: 'DecisionTree'}
    stackPredTrain = stackPredTrain.rename(columns=columns)
    stackPredTest = stackPredTest.rename(columns=columns)
    print stackMSE.rename(columns=columns)

    '''Stack: level-2, stack predictions from level-1 models. Here we exemplify it using KernelRidge.'''

    krr = KernelRidge(alpha=1, kernel='polynomial', degree=2, coef0=5)
    krr.fit(stackPredTrain, y_train_base)
    y_stack2_pred = krr.predict(stackPredTest)
    stack2Sub = pd.DataFrame(data=data_test.Id)
    stack2Sub['SalePrice'] = y_stack2_pred
    stack2Sub.to_csv('stack2Sub.csv', index=False)  # prediction for submission    