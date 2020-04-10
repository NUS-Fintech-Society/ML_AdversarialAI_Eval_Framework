import sys, os
import csv
import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
import pickle

#cleaning of data
class dataClean:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self #does nothing

    def transform(self, X):
        output = X.copy()
        output = output.drop(['home_ownership', 'income_category','term', 'application_type','purpose','interest_payments','loan_condition'], axis = 1)
        output['issue_d'] = output['issue_d'].str.replace(r'\D', '')
        labelencoder = LabelEncoder()
        output['region'] = labelencoder.fit_transform(output['region'])
        return output
            
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


#Burota transform
class BurotaTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y = None):
        return self #does nothing
    
    def transform(self, X, y = None):
        output = X.copy()
        burota_selector = pickle.load(open('BorutaFeatureSelection.pkl','rb'))
        output = burota_selector.transform(X.values)
        return output
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    


def read_csv(csvfile):
    print('read_csv(): type(csvfile)) = {}'.format(csvfile))
    print('')

    foo_df = pd.read_csv(csvfile)

    return foo_df

class DataTransform:
    def __init__(self):
        pass  

    def transform(self,x): 
        data_Clean = dataClean()
        burota_selection = BurotaTransform()
        #prediction
        data = data_Clean.transform(x)
        data = burota_selection.transform(data)
        return data
