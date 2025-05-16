import torch
from abc import ABC
import pandas as pd
import numpy as np
from ..ct_utils import eval_metrics, eval_models
import pandas as pd
import numpy as np
from ctgan.data_transformer import DataTransformer
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis, cdist
from scipy.stats import ks_2samp, wasserstein_distance

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
        
    def get_metadata(self):
        return self.metadata

    def save(self, path):
        model_data = {
            "model": self,
            "metadata": self.metadata
        }
        torch.save(model_data, path)

    def load(self, path):
        model_data = torch.load(path, weights_only=False)
        model = model_data['model']
        metadata = model_data['metadata']
        return model, metadata
    
    def eval_ml(self, real_data: pd.DataFrame, target_name: str, task: str, 
                model: BaseEstimator, metrics: list, test_size: float = 0.3, fake_data: pd.DataFrame = None):
    
        if task != "classification" and task != "regression":
            raise ValueError("Unknown task. Please choose between 'classification'/'regression'")
        if task == "classification" and not isinstance(model, ClassifierMixin):
            raise ValueError("The provided model is not a classifier, but task='classification' was specified.")
        if task == "regression" and not isinstance(model, RegressorMixin):
            raise ValueError("The provided model is not a regressor, but task='regression' was specified.")

        df = real_data.copy()
        oh_encoder = None
        if (target_name in self.discrete_columns
            and not np.issubdtype(df[target_name].dtype, np.number)):
            oh_encoder = OneHotEncoder()
            df[target_name] = oh_encoder.fit_transform(df[target_name])

        real_y = df[target_name].values
        real_X = df.drop(columns=[target_name])

        cat_cols = [c for c in self.discrete_columns if c != target_name and c in real_X]

        real_ohe = pd.get_dummies(real_X, columns=cat_cols)

        if fake_data is None:
            fake_data = self.sample(len(real_ohe))
        fake_df = fake_data.copy()
        if oh_encoder is not None:
            fake_df[target_name] = oh_encoder.transform(fake_df[target_name])
        fake_y = fake_df[target_name].values
        fake_X = fake_df.drop(columns=[target_name])
        fake_ohe = pd.get_dummies(fake_X, columns=cat_cols)

        all_cols = real_ohe.columns.union(fake_ohe.columns)
        real_ohe = real_ohe.reindex(columns=all_cols, fill_value=0)
        fake_ohe = fake_ohe.reindex(columns=all_cols, fill_value=0)

        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            real_ohe.values,
            real_y,
            test_size=test_size,
            stratify=(real_y if task=="classification" else None),
        )

        X_train_fake, y_train_fake = fake_ohe.values, fake_y

        model.fit(X_train_real, y_train_real)
        y_pred_real = model.predict(X_test_real)

        model.fit(X_train_fake, y_train_fake)
        y_pred_fake = model.predict(X_test_real)
        
        metric_results = {"real": {}, "fake": {}}
        for metric in metrics:
            if metric in eval_metrics[task]:
                if metric == "classification_report":

                    metric_results["real"][metric] = pd.DataFrame(eval_metrics[task][metric](y_test_real, y_pred_real, output_dict=True, zero_division=0)).T
                    metric_results["fake"][metric] = pd.DataFrame(eval_metrics[task][metric](y_test_real, y_pred_fake, output_dict=True, zero_division=0)).T
                else:
                    metric_results["real"][metric] = eval_metrics[task][metric](y_test_real, y_pred_real)
                    metric_results["fake"][metric] = eval_metrics[task][metric](y_test_real, y_pred_fake)

        return metric_results
    
    

    def eval_stat(self, real_data: pd.DataFrame, test: str, fake_data: pd.DataFrame = None, classifier=None):
        oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        real_data_encoded = real_data.copy()
        fake_data_encoded = fake_data.copy() if fake_data is not None else None
        real_data_encoded[self.discrete_columns] = oh_encoder.fit_transform(real_data[self.discrete_columns])
        if fake_data is not None:
            fake_data_encoded[self.discrete_columns] = oh_encoder.transform(fake_data[self.discrete_columns])
        if fake_data_encoded is None:
            n = real_data_encoded.shape[0]
            fake_data_encoded = np.array(self.transformer.transform(self.sample(n)))
            fake_data_encoded = pd.DataFrame(fake_data_encoded, columns=real_data_encoded.columns)

        test_funcs = {
            "mahalanobis": self._mahalanobis_test,
            "ks": self._ks_test,
            "wasserstein_distance": self._wasserstein_test,
            "energy_distance": self._energy_distance_test,
            "two_sample_classifier": self._two_sample_classifier_test,
        }

        if test not in test_funcs:
            raise ValueError(f"Unknown test '{test}'. Available: {list(test_funcs)}")

        return test_funcs[test](real_data_encoded, fake_data_encoded, classifier)




    ##### PRIVATE FUNCTIONS #####
    def _transform_data(self, data, discrete_columns):
        transformer = DataTransformer()
        transformer.fit(data, discrete_columns)
        self.transformer = transformer
        return transformer.transform(data)
    
    def _create_table_metadata(self, data):
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


        if self.metadata["table"]["correlations"] == {}:
            self.metadata["table"]["correlations"] = data[self.cont_columns].corr().to_dict()

    def _mahalanobis_test(self, real: pd.DataFrame, fake: pd.DataFrame) -> pd.DataFrame:
        cov = np.cov(real.values, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        distance = mahalanobis(real.mean().values, fake.mean().values, inv_cov)
        return pd.DataFrame([{'statistic': 'mahalanobis', 'value': distance.item()}])

    def _ks_test(self, real: pd.DataFrame, fake: pd.DataFrame) -> pd.DataFrame:
        results = []
        for col in real.columns:
            stat, p_value = ks_2samp(real[col], fake[col])
            results.append({
                'feature': col,
                'statistic': 'ks',
                'value': stat.item(),
                'p_value': p_value.item()
            })
        return pd.DataFrame(results)

    def _wasserstein_test(self, real: pd.DataFrame, fake: pd.DataFrame) -> pd.DataFrame:
        results = []
        for col in real.columns:
            dist = wasserstein_distance(real[col], fake[col])
            results.append({
                'feature': col,
                'statistic': 'wasserstein',
                'value': dist
            })
        return pd.DataFrame(results)


    def _energy_distance_test(self, real: pd.DataFrame, fake: pd.DataFrame) -> pd.DataFrame:
        a = cdist(real.values, real.values).mean()
        b = cdist(fake.values, fake.values).mean()
        c = cdist(real.values, fake.values).mean()
        energy_dist = 2 * c - a - b
        return pd.DataFrame([{'statistic': 'energy', 'value': energy_dist}])

    
    def _two_sample_classifier_test(self, real: pd.DataFrame, fake: pd.DataFrame, classifier: ClassifierMixin) -> pd.DataFrame:
        if classifier is None or not isinstance(classifier, ClassifierMixin):
            raise ValueError("A valid scikit-learn classifier instance must be provided.")

        X = pd.concat([real, fake], axis=0).values
        y = np.concatenate([
            np.ones(len(real)), 
            np.zeros(len(fake))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = self.eval_metrics['accuracy_score'](y_test, y_pred)
        report_text = self.eval_metrics['classification_report'](y_test, y_pred)

        return pd.DataFrame([{
            'statistic': 'two_sample_classifier_accuracy',
            'value': accuracy,
            'report': report_text
        }])
