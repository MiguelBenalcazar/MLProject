import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import train, Dataset, early_stopping

class testImputate():
    def __init__(self, data, listCategory=None, listCategoryPredictedAsRegression = None, Verbose = False) -> None:
        self.data = data.copy()
        self.listCategory = listCategory
        self.listCategoryPredictedAsRegression = listCategoryPredictedAsRegression
        self.Verbose = Verbose
    
    def verboseFunc(self):
        if self.Verbose:
            return 0
        else:
            return -1

    def __parameters__(self, isCategory, categories = None, seed = 123):
        verbose = self.verboseFunc()
        if not isCategory:
            return {
                "boosting": "random_forest",
                # "num_iterations": 2000,
                "metric": 'rmse',
                "max_depth": 10,
                "num_leaves": 128,
                "min_data_in_leaf": 1,
                "min_sum_hessian_in_leaf": 0.00001,
                "min_gain_to_split": 0.0,
                "bagging_fraction": 0.632,
                "feature_fraction": 1.0,
                "feature_fraction_bynode": 0.632,
                "bagging_freq": 10,
                "seed":seed,
                "verbosity": verbose,
    
            }
        else:
            return {
                'objective': 'multiclassova',
                'num_class': len(categories) + 1,  # Specify the number of classes
                'metric': 'multi_error',
                'boosting_type': 'random_forest',
                "max_depth": 10,
                'num_leaves': 128,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                "seed":seed,
                'verbose': verbose
                }
        
    # def __parameters__(self, isCategory, categories = None):
    #     if not isCategory:
    #         return {
    #             "objective":"regression",
    #             "boosting": "random_forest", 
    #             "num_iterations": 128,
    #             "max_depth": 8,
    #             "num_leaves": 128,
    #             "min_data_in_leaf": 1,
    #             "min_sum_hessian_in_leaf": 0.00001,
    #             "min_gain_to_split": 0.0,
    #             "bagging_fraction": 0.632,
    #             "feature_fraction": 1.0,
    #             "feature_fraction_bynode": 0.632,
    #             "bagging_freq": 1,
    #             "verbosity": -1
    #             }
    #     else:
    #         return {
    #             'objective': 'multiclass',
    #             'num_class': len(categories) +1,  # Specify the number of classes
    #             'metric': 'multi_logloss',
    #             'boosting_type': 'gbdt',
    #             'num_leaves': 128,
    #             'learning_rate': 0.05,
    #             'feature_fraction': 0.9,
    #             'bagging_fraction': 0.8,
    #             'bagging_freq': 5,
    #             'verbose': -1
    #             }


    def miceProcessNotCategory(self, n, savePredictions):
        for feature in self.data.columns:
            self.models['features'][feature]={}

        for feature in self.data.columns:
            self.models['features'][feature]['models']= []
            TrainTotalScaled_X, TrainTotal_y, X_predPD = self.preprocessing(feature)
            prediction = pd.DataFrame()
            for i in range(n):
                X_train, X_validation, y_train, y_validation = self.splitDataset(TrainTotalScaled_X, TrainTotal_y, randomState = i)
                prediction[f'pred_{i}'], modelT = self.__model__(X_train, X_validation, y_train, y_validation, X_predPD, seed=i)
                self.models['features'][feature]['models'].append(modelT)
              
            newImpute = prediction.mean(axis=1)
            prediction['prediction'] = newImpute

            if self.listCategoryPredictedAsRegression != None:
                if feature in self.listCategoryPredictedAsRegression:
                    prediction = prediction.astype(int)


            prediction['id'] = X_predPD.index
            prediction = prediction.set_index('id')

            if savePredictions:
                self.predictions[feature] = prediction

            self.data[feature].update(prediction['prediction'] )

    def miceProcessCategory(self, n , savePredictions):
        for feature in self.data.columns:
            self.models['features'][feature]={}

        for feature in self.data.columns:
            self.models['features'][feature]['models']= []
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
                    prediction[f'pred_{i}'], modelT = self.__model__(X_train, X_validation, y_train, y_validation, X_predPD, seed=i)
                    # keep model to save
                    self.models['features'][feature]['models'].append(modelT)
                else:
                    prediction[f'pred_{i}'], modelT = self.__modelCat__(X_train, X_validation, y_train, y_validation, X_predPD, categories=cat, seed=i)
                    # keep model to save
                    self.models['features'][feature]['models'].append(modelT)

            if not isCategory:
                newImpute = prediction.mean(axis=1)
                prediction['prediction'] = newImpute

                if self.listCategoryPredictedAsRegression != None:
                    if feature in self.listCategoryPredictedAsRegression:
                        prediction = prediction.astype(int)
                
                prediction['id'] = X_predPD.index
                prediction = prediction.set_index('id')

                if savePredictions:
                    self.predictions[feature] = prediction
                
                self.data[feature].update(prediction['prediction'] )
            else:
                newImpute = prediction.mode(axis=1).iloc[:, 0]
                prediction['prediction'] = newImpute

                prediction['id'] = X_predPD.index
                prediction = prediction.set_index('id')

                if savePredictions:
                    self.predictions[feature] = prediction
                
                self.data[feature].update(prediction['prediction'] )

       


    def mice(self, n=1, savePredictions = False):
        self.models = {
            'listCategory': self.listCategory,
            'listCategoryPredictedAsRegression': self.listCategoryPredictedAsRegression,
            'features':{} 
        }

        if savePredictions:
            self.predictions = {}

        if self.listCategory == None:
            self.miceProcessNotCategory(n, savePredictions)
        else:
            self.miceProcessCategory(n , savePredictions)

        
        print("-- Imputation Done --")



    def __model__(self, X_train, X_validation, y_train, y_validation, X_predPD, seed =123):
        
        
        train_data = Dataset(X_train, label=y_train)
        validation_data = Dataset(X_validation, label=y_validation)
        bst = train(self.__parameters__(False, seed=seed), train_data, valid_sets=[validation_data],  callbacks=[early_stopping(stopping_rounds=5)])
        # bst = train(self.__parameters__(False), train_data, valid_sets=[validation_data])
        return bst.predict(X_predPD), bst
    
    def __modelCat__(self, X_train, X_validation, y_train, y_validation, X_predPD, categories=None, seed =123):
        train_data = Dataset(X_train, label=y_train)
        validation_data = Dataset(X_validation, label=y_validation)
        bst = train(self.__parameters__(True, categories, seed=seed), train_data, valid_sets=[validation_data], callbacks=[early_stopping(stopping_rounds=5)])
        pred = bst.predict(X_predPD)
        y_pred = pred.argmax(axis=1)
        return y_pred , bst


    def miceTrainedModel(self, dataset, savePredictions):
        try:
            # Try to access self.models
            model = self.models

            #Vefiry features
            newDataKeys = dataset.keys()
            for i in self.models['features'].keys():
                if i not in newDataKeys:
                    raise ValueError(f'{i} Does not exist in the current data')
 
            

            if savePredictions:
                self.predictionsNewData = {}

            if model['listCategory'] == None:
                dataset = self.miceProcessNotCategoryNewData(dataset, savePredictions)
             
            else:
                dataset = self.miceProcessCategoryNewData(dataset , savePredictions)
               
            print("-- Imputation Done --")

            return dataset

            
        except AttributeError:
            print("self.models does not exist")

    

    def miceProcessNotCategoryNewData(self, data, savePredictions):
        for feature in data.columns:
            _, _, X_predPD = self.preprocessingNewData(data,  feature)
            prediction = pd.DataFrame()

            for i, model in enumerate(self.models['features'][feature]['models']):

                prediction[f'pred_{i}'] = self.__modelNewData__(model, X_predPD)

            newImpute = prediction.mean(axis=1)
            prediction['prediction'] = newImpute

            if self.listCategoryPredictedAsRegression != None:
                if feature in self.listCategoryPredictedAsRegression:
                    prediction = prediction.astype(int)


            prediction['id'] = X_predPD.index
            prediction = prediction.set_index('id')

            if savePredictions:
                self.predictionsNewData[feature] = prediction

            return data[feature].update(prediction['prediction'] )   

    def miceProcessCategoryNewData(self, data , savePredictions):
        if not data.isnull().values.any():
            raise ValueError('Data does not have nan values')

        for feature in data.columns:

 
            isCategory = False
            _, _, X_predPD = self.preprocessingNewData(data, feature)
            prediction = pd.DataFrame()
            
            if feature in self.models['listCategory']:
                isCategory = True
          
             
            prediction = pd.DataFrame()

            for i, model in enumerate(self.models['features'][feature]['models']):
                if not isCategory:
                    prediction[f'pred_{i}'] = self.__modelNewData__(model, X_predPD)
                else:
                    prediction[f'pred_{i}'] = self.__modelCatNewData__(model, X_predPD)
               

            if not isCategory:
                newImpute = prediction.mean(axis=1)
                prediction['prediction'] = newImpute

                if self.listCategoryPredictedAsRegression != None:
                    if feature in self.listCategoryPredictedAsRegression:
                        prediction = prediction.astype(int)
                
                prediction['id'] = X_predPD.index
                prediction = prediction.set_index('id')

                if savePredictions:
                    self.predictionsNewData[feature] = prediction
                
                data[feature].update(prediction['prediction'] )
                

            else:
                newImpute = prediction.mode(axis=1).iloc[:, 0]
                prediction['prediction'] = newImpute

                prediction['id'] = X_predPD.index
                prediction = prediction.set_index('id')

                if savePredictions:
                    self.predictionsNewData[feature] = prediction
                
                data[feature].update(prediction['prediction'] )
                
        return data
                


    
    
    def __modelNewData__(self, model, X_predPD):
        return model.predict(X_predPD)
    
    def __modelCatNewData__(self, model, X_predPD):
        pred = model.predict(X_predPD)
        y_pred = pred.argmax(axis=1)
        return y_pred



    
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

        # keep scaler per feature
        self.models['features'][feature]['scaler'] = scaler
        

        # Fixed ID
        TrainTotalScaled_X['id'] = TrainTotal_y.index
        TrainTotalScaled_X = TrainTotalScaled_X.set_index('id')

        # Prediceted Data Scaler
        X_pred =  scaler.transform(predictData)
        X_predPD = pd.DataFrame(X_pred, columns=predictData.columns)

        X_predPD['id'] = predictData.index
        X_predPD = X_predPD.set_index('id')

        return TrainTotalScaled_X, TrainTotal_y, X_predPD
    
    def scaleData(self, data):
        scaler = StandardScaler()
        scaledData = scaler.fit_transform(data)
        scaledDataPD = pd.DataFrame(scaledData, columns=data.columns)
        return scaler, scaledDataPD
    
    def preprocessingNewData(self, data, feature):
        print(f"___ Processing {feature} ___")
        auxDataCopy = data.copy()
        # Delete nan data from data to predict
        featurePredict = auxDataCopy[auxDataCopy[feature].isna()]
        # Data to test model
        predictData = featurePredict.loc[:, ~featurePredict.columns.isin([feature])]
        trainData = auxDataCopy.dropna(axis=0, how="any", subset=[feature])
        # Create data to train
        TrainTotal_y = trainData.loc[:, trainData.columns.isin([feature])]
        TrainTotal_X = trainData.loc[:, ~trainData.columns.isin([feature])]
        scaler, TrainTotalScaled_X = self.scaleDataNewData(TrainTotal_X, feature)

        # Fixed ID
        TrainTotalScaled_X['id'] = TrainTotal_y.index
        TrainTotalScaled_X = TrainTotalScaled_X.set_index('id')

        # Prediceted Data Scaler
        X_pred =  scaler.transform(predictData)
        X_predPD = pd.DataFrame(X_pred, columns=predictData.columns)

        X_predPD['id'] = predictData.index
        X_predPD = X_predPD.set_index('id')

        return TrainTotalScaled_X, TrainTotal_y, X_predPD
    
    def scaleDataNewData(self, data, feature):
        scaler = self.models['features'][feature]['scaler']
        scaledData = scaler.transform(data)
        scaledDataPD = pd.DataFrame(scaledData, columns=data.columns)
        return scaler, scaledDataPD
    
    def plotCorrelation(self, data = None):
        fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

        if data == None:
            data = self.data

      
        mask = np.triu(np.ones_like(data.corr()))
        sns.heatmap(data.corr(), cmap="coolwarm", annot=True, mask=mask)
        
        
        

    def savePredictions(self, filename='obj'):
        import pickle

        # Save the object to a file
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(self.models, f)

    

    def loadPredictions(self, filename='obj'):
        import pickle

        # Load the object from the file
        with open(f"{filename}.pkl", "rb") as f:
            self.models = pickle.load(f)

        

        