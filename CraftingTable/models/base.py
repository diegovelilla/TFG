from abc import ABC, abstractmethod
from ..utils import eval_metrics
from ctgan.data_transformer import DataTransformer
import torch
import os.path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis, cdist
from scipy.stats import ks_2samp, wasserstein_distance, combine_pvalues

# Base class for every other model
class BaseModel(ABC):

    def __init__(self):
        self.metadata = {
            "table": {
                "columns": {

                },
                "correlations": {
                    
                },
                
            },
            "model":{
                "model_type": "",
                "hyperparameters": {

                },
                "fit_settings": {
                    "times_fitted": 0,
                    "fit_history": [
                        
                    ]
                }
            }
        }
    
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass
        
    def get_metadata(self) -> dict:
        return self.metadata

    def save(self, path: str):
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")

        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path) -> "BaseModel":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at {path}")
        
        obj = cls.__new__(cls)
        obj.__dict__.update(torch.load(path, weights_only=False))
        return obj
    
    def eval_ml(self, real_data: pd.DataFrame, target_name: str, task: str, 
                model: BaseEstimator, metrics: list, test_size: float = 0.3, fake_data: pd.DataFrame = None) -> dict:
    
        if self.metadata['model']['fit_settings']['times_fitted'] == 0:
            raise RuntimeError("Model is not yet fitted. Please fit the model before evaluation.")
        if not isinstance(real_data, pd.DataFrame):
            raise TypeError("real_data must be a pandas DataFrame.")
        if not isinstance(target_name, str):
            raise TypeError("target_name must be a string.")
        if task != "classification" and task != "regression":
            raise ValueError("Unknown task. Please choose between 'classification'/'regression'")
        if not isinstance(metrics, list):
            raise TypeError("metrics must be a list of strings.")
        if task == "classification" and not isinstance(model, ClassifierMixin):
            raise ValueError("The provided model is not a classifier, but task='classification' was specified.")
        if task == "regression" and not isinstance(model, RegressorMixin):
            raise ValueError("The provided model is not a regressor, but task='regression' was specified.")

        df = real_data.copy()

        y = df[target_name]
        X = df.drop(columns=[target_name])

        cat_cols = [col for col in self.discrete_columns if col != target_name and col in X.columns]
        X_encoded = pd.get_dummies(X, columns=cat_cols)

        if not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
        else:
            label_encoder = None
            y_encoded = y.values

        if fake_data is None:
            fake_data = self.sample(len(X_encoded))

        fake_df = fake_data.copy()
        fake_y = fake_df[target_name]
        fake_X = fake_df.drop(columns=[target_name])

        fake_X_encoded = pd.get_dummies(fake_X, columns=cat_cols)

        all_cols = X_encoded.columns.union(fake_X_encoded.columns)
        X_encoded = X_encoded.reindex(columns=all_cols, fill_value=0)
        fake_X_encoded = fake_X_encoded.reindex(columns=all_cols, fill_value=0)

        if label_encoder is not None:
            fake_y_encoded = label_encoder.transform(fake_y)
        else:
            fake_y_encoded = fake_y.values

        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_encoded.values,
            y_encoded,
            test_size=test_size,
            stratify=(y_encoded if task == "classification" else None),
        )

        X_train_fake = fake_X_encoded.values
        y_train_fake = fake_y_encoded

        model.fit(X_train_real, y_train_real)
        y_pred_real = model.predict(X_test_real)

        model.fit(X_train_fake, y_train_fake)
        y_pred_fake = model.predict(X_test_real)

        metric_results = {"real": {}, "fake": {}}
        for metric in metrics:
            if metric in eval_metrics[task]:
                if metric == "classification_report":
                    metric_results["real"][metric] = pd.DataFrame(
                        eval_metrics[task][metric](y_test_real, y_pred_real, output_dict=True, zero_division=0)
                    ).T
                    metric_results["fake"][metric] = pd.DataFrame(
                        eval_metrics[task][metric](y_test_real, y_pred_fake, output_dict=True, zero_division=0)
                    ).T
                else:
                    metric_results["real"][metric] = eval_metrics[task][metric](y_test_real, y_pred_real)
                    metric_results["fake"][metric] = eval_metrics[task][metric](y_test_real, y_pred_fake)

        return metric_results
    

    def eval_stat(self, real_data: pd.DataFrame, test: str, fake_data: pd.DataFrame = None, classifier=None) -> dict:
        test_funcs = {
            "mahalanobis": self._mahalanobis_test,
            "ks": self._ks_test,
            "wasserstein_distance": self._wasserstein_test,
            "energy_distance": self._energy_distance_test,
            "two_sample_classifier": self._two_sample_classifier_test,
        }

        if test not in test_funcs:
            raise ValueError(f"Unknown test '{test}'. Available: {list(test_funcs)}")
        if test != "two_sample_classifier":
            if classifier is None or not isinstance(classifier, ClassifierMixin):
                raise ValueError("A valid scikit-learn classifier instance must be provided.")
        if not isinstance(real_data, pd.DataFrame):
            raise TypeError("real must be a pandas DataFrame.")
        if not fake_data and not isinstance(fake_data, pd.DataFrame):
            raise TypeError("fake must be a pandas DataFrame.")
        if real_data.shape[1] != fake_data.shape[1]:
            raise ValueError("Real and fake datasets must have the same number of features.")
            
        oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        real_data_encoded = real_data.copy()
        real_cat_encoded = oh_encoder.fit_transform(real_data_encoded[self.discrete_columns])
        real_cat_columns = oh_encoder.get_feature_names_out(self.discrete_columns)
        real_data_encoded = real_data_encoded.drop(columns=self.discrete_columns).reset_index(drop=True)
        real_data_encoded = pd.concat([real_data_encoded, pd.DataFrame(real_cat_encoded, columns=real_cat_columns)], axis=1)
        
        if fake_data is None:
            print("No fake data provided, sampling from the model...")
            n = real_data_encoded.shape[0]
            fake_data = self.sample(n)
            print(f"Sampled {n} rows of fake data.")
        fake_data_encoded = fake_data.copy()

        fake_cat_encoded = oh_encoder.transform(fake_data_encoded[self.discrete_columns])
        fake_cat_columns = oh_encoder.get_feature_names_out(self.discrete_columns)
        fake_data_encoded = fake_data_encoded.drop(columns=self.discrete_columns).reset_index(drop=True)
        fake_data_encoded = pd.concat([fake_data_encoded, pd.DataFrame(fake_cat_encoded, columns=fake_cat_columns)], axis=1)
    
        if test == "two_sample_classifier":
            return test_funcs[test](real_data_encoded, fake_data_encoded, classifier)
        
        return test_funcs[test](real_data_encoded, fake_data_encoded)


    ##### PRIVATE FUNCTIONS #####
    def _transform_data(self, data: pd.DataFrame, discrete_columns: list[str]) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(discrete_columns, list[str]):
            raise TypeError("discrete_columns must be a list of column names.")
        transformer = DataTransformer()
        transformer.fit(data, discrete_columns)
        self.transformer = transformer
        return transformer.transform(data)
    
    def _create_table_metadata(self, data: pd.DataFrame):
        if self.metadata["table"]["columns"] == {}:
            for column in data.columns:
                if column in self.cont_columns:
                    column_dict = {
                        "dtype": str(data[column].dtype),
                        "max": np.max(data[column]).item(),
                        "min": np.min(data[column]).item(),
                        "avg": np.average(data[column]).item(),
                        "std": np.std(data[column]).item(),
                        "median": np.median(data[column]).item(),
                    }
                else:
                    column_dict = {
                        "dtype": str(data[column].dtype),
                        "mode": data[column].mode().iloc[0],
                        "nunique": data[column].nunique(),
                        "value_counts": data[column].value_counts().to_dict(),
                    }
                self.metadata["table"]["columns"][column] = column_dict
            self.metadata["table"]["columns"]["discrete_columns"] = self.discrete_columns
            self.metadata["table"]["columns"]["continous_columns"] = self.cont_columns

        if self.metadata["table"]["correlations"] == {}:
            self.metadata["table"]["correlations"] = data[self.cont_columns].corr().to_dict()

    def _mahalanobis_test(self, real: pd.DataFrame, fake: pd.DataFrame) -> dict:
        cov = np.cov(real.values, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        distance = mahalanobis(real.mean().values, fake.mean().values, inv_cov)
        return {'statistic': 'mahalanobis', 'value': distance.item()}

    def _ks_test(self, real: pd.DataFrame, fake: pd.DataFrame) -> dict:
        results = []
        p_values = []

        for col in real.columns:
            stat, p_value = ks_2samp(real[col], fake[col])
            p_values.append(p_value)
            results.append({
                'feature': col,
                'value': stat.item(),
                'p_value': p_value.item()
            })

        fisher_stat, fisher_p = combine_pvalues(p_values, method='fisher')
        return {
            'statistic': 'ks',
            'value': results,
            'global': {
                'method': 'fisher',
                'statistic': fisher_stat.item(),
                'p_value': fisher_p.item()
            }
        }

    def _wasserstein_test(self, real: pd.DataFrame, fake: pd.DataFrame) -> dict:
        results = []
        for col in real.columns:
            dist = wasserstein_distance(real[col], fake[col])
            results.append({
                'feature': col,
                'value': dist
            })
        return {'statistic': 'wasserstein', "value": results}


    def _energy_distance_test(self, real: pd.DataFrame, fake: pd.DataFrame) -> dict:
        print(fake.shape, real.shape)
        a = cdist(real.values, real.values).mean()
        b = cdist(fake.values, fake.values).mean()
        c = cdist(real.values, fake.values).mean()
        energy_dist = 2 * c - a - b
        return {'statistic': 'energy', 'value': energy_dist}

    
    def _two_sample_classifier_test(self, real: pd.DataFrame, fake: pd.DataFrame, classifier: ClassifierMixin) -> dict:
        X = pd.concat([real, fake], axis=0).values
        y = np.concatenate([
            np.ones(len(real)), 
            np.zeros(len(fake))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = eval_metrics['classification']['accuracy'](y_test, y_pred)
        report_text = eval_metrics['classification']['classification_report'](y_test, y_pred)

        return {
            'statistic': 'two_sample_classifier_accuracy',
            'value': accuracy,
            'report': report_text
        }
