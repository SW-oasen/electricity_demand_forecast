# Split the data into training and testing sets based on the date
def train_test_split_by_date(df, date_column, target_column, split_date):
    train_data = df[df[date_column] < split_date]
    test_data = df[df[date_column] >= split_date]
    features_train = train_data.drop([date_column, target_column], axis=1)
    target_train = train_data[target_column]
    features_test = test_data.drop([date_column, target_column], axis=1)
    target_test = test_data[target_column]
    return features_train, target_train, features_test, target_test

def train_test_split_by_date_for_sarimax(df, date_column, target_column, split_date):
    train_data = df[df[date_column] < split_date]
    test_data = df[df[date_column] >= split_date]

    features_train = train_data.drop(target_column, axis=1)
    features_test = test_data.drop(target_column, axis=1)
    target_train = train_data[[date_column, target_column]]
    target_test = test_data[[date_column, target_column]]

    return features_train, target_train, features_test, target_test

# column transformer pipeline for preprocessing
from sklearn.compose import ColumnTransformer   
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def init_preprocessor(in_df):
    # identify numeric and categorical columns
    numeric_features = in_df.select_dtypes(include=['float64', 'float32']).columns
    categorical_features = in_df.select_dtypes(include=['int64', 'int32']).columns
    # numeric transformer
    numeric_transformer = Pipeline(steps=[ 
        ('scaler', StandardScaler()) 
    ])  
    # categorical transformer
    categorical_transformer = Pipeline(steps=[  
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])  

    # combine transformers into a column transformer
    preprocessor = ColumnTransformer(   
        transformers=[ 
            ('num', numeric_transformer, numeric_features), 
            ('cat', categorical_transformer, categorical_features) 
        ])
    return preprocessor


# build the model pipeline
def init_model_pipeline(in_df, model):
    preprocessor = init_preprocessor(in_df)
    model_pipeline = Pipeline(steps=[ 
        ('preprocessor', preprocessor), 
        ('model', model) 
    ])  
    return model_pipeline

def train_model_predict(model_pipeline, features_train, target_train, features_future):
    model_pipeline.fit(features_train, target_train)
    predictions = model_pipeline.predict(features_future)
    return predictions

import numpy as np

def print_scores(model_name, target_test, predictions):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(target_test, predictions)
    rmse = np.sqrt(mean_squared_error(target_test, predictions))
    r2 = r2_score(target_test, predictions)
    print("-"*20 + f" scoring " + "-"*20)
    print(f"{'model':<15} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print(f'{model_name:<16} {mae:>10.2f} {rmse:>10.2f} {r2:>8.2f}')
    

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, learning_curve, learning_curve

# tune the model using GridSearchCV
def tune_model_grid(model_pipeline, 
                    in_param_grid, 
                    in_features_train, 
                    in_target_train):
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=model_pipeline, 
                               param_grid=in_param_grid, 
                               cv=tscv,
                               scoring='neg_mean_absolute_error', 
                               n_jobs=-1)
    grid_search.fit(in_features_train, in_target_train)
    #print(f'Best parameters: {grid_search.best_params_}')
    return grid_search.best_estimator_, grid_search.best_params_

try:
    from skopt import BayesSearchCV
except ImportError:
    BayesSearchCV = None

# tune the model using Bayesian optimization
def tune_model_bayesian(model_pipeline, 
                        in_param_bayes, 
                        in_features_train, 
                        in_target_train):
    tscv = TimeSeriesSplit(n_splits=5)
    bayes_search = BayesSearchCV(estimator=model_pipeline, 
                                 search_spaces=in_param_bayes, 
                                 cv=tscv,
                                 scoring='neg_mean_absolute_error', 
                                 n_jobs=-1)
    bayes_search.fit(in_features_train, in_target_train)
    #print(f'Best parameters: {bayes_search.best_params_}')
    return bayes_search.best_estimator_, bayes_search.best_params_

import pickle

# save trained model to pickle file
def save_model_to_pickle(model_pipeline, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model_pipeline, f)

# load trained model from pickle file
def load_model_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        model_pipeline = pickle.load(f)
    return model_pipeline

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# learn curve for training and testing data
def plot_learning_curve(model_pipeline, model_name, features_train, target_train):
    tscv = TimeSeriesSplit(n_splits=3)
    train_sizes, train_scores, test_scores = learning_curve(model_pipeline, 
                                                            features_train, 
                                                            target_train, 
                                                            cv=tscv, 
                                                            scoring='neg_mean_absolute_error', n_jobs=-1)
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(6, 3))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='orange', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='steelblue', label='Cross-validation score')
    plt.title(f'Learning Curve of {model_name}')
    plt.xlabel('Training examples')
    plt.ylabel('MAE')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
