# XGBoost Model
With reference to research models on Predicting Credit Risk, we have found that using a XGBoost model with Burota algorithm feature selction
is one of the most popular model out there.

## Usage

### Jupyter notebook for forming model
The `XGBoost_model.ipynb` jupyter file contains all the workings and training of models and algorithm to predict credit risk data. It also includes explanations on feature selection and data cleaning.

### Pickled model modules
`XGBoostModel.pkl` and `BorutaFeatureSelection.pkl` files are pickled saved model modules which are used in `XGBoost_Predictor.py` python file which contains the model

### Python file containing saved model
The `XGBoost_Predictor.py` file is a python file that contains the saved model that can be run to predict values. usage will be on command line, where the format is as shown:
`$ python XGBoost_Predictor.py <input csv to be predicted> --accuracy <actual csv file for target values>`
The output on command line will show the accuracy of predicted values by the model

### Sample Datasets
In the `sample datasets` folder, it contains sample datasets which is extracted from `Credit_Risk_Data` dataset. The dataset contains 20,000 rows and contains `x_test.csv` file which contains values for the model to predict, and `y_test.csv` file that contains actual target values for computation of model accuracy.
