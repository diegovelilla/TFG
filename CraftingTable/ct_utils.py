import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

model_hyperparameters = {
            "ctgan":{
                "embedding_dim": 128,
                "generator_dim": (256, 256),
                "discriminator_dim": (256, 256),
                "generator_lr": 2e-4,
                "generator_decay": 1e-6,
                "discriminator_lr": 2e-4,
                "discriminator_decay": 1e-6,
                "batch_size": 500,
                "discriminator_steps": 1,
                "log_frequency": 1,
                "verbose": False,
                "epochs": 300,
                "pac": 10,
                "cuda": True
            },
            "tvae":{
                "embedding_dim": 128,
                "compress_dims": (128, 128),
                "decompress_dims": (128, 128),
                "l2scale": 1e-5,
                "batch_size": 500,
                "epochs": 300,
                "cuda": True,
                "verbose": False
            },
            "diffusion":{
                "d_layers": [128],
                "device": torch.device('cuda:0'), 
                "seed":0
            }}

eval_models = {
            "classification": {
                            "LogisticRegression": LogisticRegression,
                            "RidgeClassifier": RidgeClassifier,
                            "RandomForestClassifier": RandomForestClassifier,
                            "GradientBoostingClassifier": GradientBoostingClassifier,
                            "SVC": SVC,
                            "DecisionTreeClassifier": DecisionTreeClassifier,
                            "KNeighborsClassifier": KNeighborsClassifier,
                            "GaussianNB": GaussianNB,
                            },
            "regression": {
                            "LinearRegression": LinearRegression,
                            "Ridge": Ridge,
                            "RandomForestRegressor": RandomForestRegressor,
                            "GradientBoostingRegressor": GradientBoostingRegressor,
                            "SVR": SVR,
                            "DecisionTreeRegressor": DecisionTreeRegressor,
                            "KNeighborsRegressor": KNeighborsRegressor,
                            }
        }

eval_metrics = {
            "classification": {
                "accuracy": accuracy_score,
                "precision": precision_score,
                "recall": recall_score,
                "f1_score": f1_score,
                "roc_auc": roc_auc_score,
                "log_loss": log_loss,
                "classification_report": classification_report
            },
            "regression": {
                "r2_score": r2_score,
                "mean_absolute_error": mean_absolute_error,
                "mean_squared_error": mean_squared_error,
                "root_mean_squared_error": root_mean_squared_error,
                "median_absolute_error": median_absolute_error,
                "explained_variance": explained_variance_score
            }
        }