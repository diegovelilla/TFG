import torch
from abc import ABC
import pandas as pd
import numpy as np
from ..ct_utils import eval_metrics, eval_models
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from scipy.stats import ks_2samp


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
                ml: str, metrics: list, test_size: float = 0.3, fake_data: pd.DataFrame = None):
    
        assert task == "classification" or task == "regression", "Unknown task. Please choose between 'classification'/'regression'"
        assert ml in eval_models[task].keys(), f"Unknown machine learning model for {task}. Please select one from the following: {eval_models[task].keys()}."


        df = real_data.copy()
        label_encoder = None
        if (target_name in self.discrete_columns
            and not np.issubdtype(df[target_name].dtype, np.number)
        ):
            label_encoder = LabelEncoder()
            df[target_name] = label_encoder.fit_transform(df[target_name])

        real_y = df[target_name].values
        real_X = df.drop(columns=[target_name])

        cat_cols = [c for c in self.discrete_columns if c != target_name and c in real_X]

        real_ohe = pd.get_dummies(real_X, columns=cat_cols)

        if fake_data is None:
            fake_data = self.sample(len(real_ohe))
        fake_df = fake_data.copy()
        if label_encoder is not None:
            fake_df[target_name] = label_encoder.transform(fake_df[target_name])
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

        model = eval_models[task][ml]()
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
    
    def eval_stat(self, real_data: pd.DataFrame, test: str, fake_data: pd.DataFrame = None, classifier: str = None):
        """
        Evaluates synthetic data quality by comparing fake samples to real data via statistical tests. 

        Args:
            real_data (pandas.DataFrame): Pandas DataFrame containing samples of real data.
            test (str): String indicating statistical test to perform.
        
        Raise: 
            ValueError: If `test` is not one of the available ones.
        """
        
        real_data_transformed = np.array(self.transformer.transform(real_data))
        if fake_data is None:
            n = real_data.shape[0] 
            fake_data = np.array(self.transformer.transform(self.sample(n)))

        if test == "mahalanobis":
            cov = np.cov(real_data_transformed, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            distance = mahalanobis(real_data_transformed.mean(axis=0), fake_data.mean(axis=0), inv_cov)
            return {'distance': distance.item()}
        
        elif test == "ks":
            ks_results = []
            for i in range(real_data_transformed.shape[1]):
                stat, p_value = ks_2samp(real_data_transformed[:, i], fake_data[:, i])
                ks_results.append((i, stat.item(), p_value.item()))
            return ks_results
    
        elif test == "two_sample_classifier":
            assert classifier is not None and classifier in eval_models['classification'].keys(), "Classifier not selected/unavailable."
            classifier_model = eval_models['classification'][classifier]()
            X = np.vstack([real_data_transformed, fake_data])
            y = np.concatenate([np.ones(real_data_transformed.shape[0]), np.zeros(fake_data.shape[0])])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            classifier_model.fit(X_train, y_train)
            y_pred = classifier_model.predict(X_test)
            accuracy = eval_metrics['accuracy_score'](y_test, y_pred)
            report = eval_metrics['classification_report'](y_test, y_pred)
            return {"accuracy": accuracy, "report": report}
            
        else:
            raise ValueError("Invalid plot type. Choose from 'ks', 'mahalanobis'.")