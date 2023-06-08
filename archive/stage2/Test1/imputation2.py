import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import train, Dataset, early_stopping

class testImputate():
    def __init__(self, data, listCategory) -> None:
        self.data = data.copy()
        self.listCategory = listCategory
        
    def __parameters__(self, isCategory, categories = None):
        if not isCategory:
            return {
                "objective":"regression",
                "boosting": "random_forest", 
                "num_iterations": 128,
                "max_depth": 8,
                "num_leaves": 128,
                "min_data_in_leaf": 1,
                "min_sum_hessian_in_leaf": 0.00001,
                "min_gain_to_split": 0.0,
                "bagging_fraction": 0.632,
                "feature_fraction": 1.0,
                "feature_fraction_bynode": 0.632,
                "bagging_freq": 1,
                "verbosity": -1
                }
        else:
            # return {
            #     'objective': 'multiclass',
            #     'num_class': len(categories) +1,  # Specify the number of classes
            #     'metric': 'multi_logloss',
            #     'boosting_type': 'gbdt',
            #     'num_leaves': 128,
            #     'learning_rate': 0.05,
            #     'feature_fraction': 0.9,
            #     'bagging_fraction': 0.8,
            #     'bagging_freq': 5,
            #     'verbose': -1
            #     }
            return {
                'objective': 'multiclass',
                'num_class': len(categories) +1,  # Specify the number of classes
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 128,
                'learning_rate': 0.001,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
                }



    def mice(self, n=1):
        self.__getDataType__()
        for feature in self.data.columns:
            isCategory = False
            TrainTotalScaled_X, TrainTotal_y, X_predPD = self.preprocessing(feature)
            # Verify if feature is in list of categories
            cat = None
            if feature in self.listCategory:
                isCategory = True
                TrainTotal_y = TrainTotal_y.astype(int)
                cat = TrainTotal_y[feature].unique()
             
            prediction = pd.DataFrame()
            for i in range(n):
                X_train, X_validation, y_train, y_validation = self.splitDataset(TrainTotalScaled_X, TrainTotal_y, randomState = i)
                if not isCategory:
                    prediction[f'pred_{i}'] = self.__model__(X_train, X_validation, y_train, y_validation, X_predPD)
                else:
                    prediction[f'pred_{i}'] = self.__modelCat__(X_train, X_validation, y_train, y_validation, X_predPD, categories=cat)
               
            if not isCategory:
                newImpute = prediction.mean(axis=1)
                prediction['prediction'] = newImpute
                
                prediction['id'] = X_predPD.index
                prediction = prediction.set_index('id')
                
                self.data[feature].update(prediction['prediction'] )
            else:
                newImpute = prediction.mode(axis=1).iloc[:, 0]
                prediction['prediction'] = newImpute

                prediction['id'] = X_predPD.index
                prediction = prediction.set_index('id')
                
                self.data[feature].update(prediction['prediction'] )
                


    def __model__(self, X_train, X_validation, y_train, y_validation, X_predPD):
        
        
        train_data = Dataset(X_train, label=y_train)
        validation_data = Dataset(X_validation, label=y_validation)
        bst = train(self.__parameters__(False), train_data, valid_sets=[validation_data], callbacks=[early_stopping(stopping_rounds=5)])
        return bst.predict(X_predPD)



    def __modelCat__(self, X_train, X_validation, y_train, y_validation, X_predPD, categories=None):
        train_data = Dataset(X_train, label=y_train)
        validation_data = Dataset(X_validation, label=y_validation)
        bst = train(self.__parameters__(True, categories), train_data, valid_sets=[validation_data], callbacks=[early_stopping(stopping_rounds=5)])
        pred = bst.predict(X_predPD)
        y_pred = pred.argmax(axis=1)
        return y_pred


    def __getDataType__(self):
        self.dataType = {}
        for i in self.data.columns:
            self.dataType[i] = self.data[i].dtype

    def scaleData(self, data):
        scaler = StandardScaler()
        scaledData = scaler.fit_transform(data)
        scaledDataPD = pd.DataFrame(scaledData, columns=data.columns)
        return scaler, scaledDataPD
    
    def splitDataset(self, dataX, dataY, randomState = 0):
        X_train, X_validation, y_train, y_validation = train_test_split(dataX, dataY, test_size=0.2, random_state=randomState)
        return X_train, X_validation, y_train, y_validation 

    def preprocessing(self, feature):
        print(f"___ Processing {feature} ___")
        auxDataCopy = self.data.copy()
        # Delete nan data from data to predict
        featurePredict = auxDataCopy[auxDataCopy[feature].isna()]
        # Data to test model
        predictData = featurePredict.loc[:, ~featurePredict.columns.isin([feature])]
        trainData = auxDataCopy.dropna(axis=0, how="any", subset=[feature])
        # Create data to train
        TrainTotal_y = trainData.loc[:, trainData.columns.isin([feature])]
        TrainTotal_X = trainData.loc[:, ~trainData.columns.isin([feature])]
        scaler, TrainTotalScaled_X = self.scaleData(TrainTotal_X)

        # Fixed ID
        TrainTotalScaled_X['id'] = TrainTotal_y.index
        TrainTotalScaled_X = TrainTotalScaled_X.set_index('id')

        # Prediceted Data Scaler
        X_pred =  scaler.transform(predictData)
        X_predPD = pd.DataFrame(X_pred, columns=predictData.columns)

        X_predPD['id'] = predictData.index
        X_predPD = X_predPD.set_index('id')

        return TrainTotalScaled_X, TrainTotal_y, X_predPD
    
    def plotCorrelation(self):
        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
        mask = np.triu(np.ones_like(self.data.corr()))
        sns.heatmap(self.data.corr(), cmap="coolwarm", annot=True, mask=mask)
        


        

        