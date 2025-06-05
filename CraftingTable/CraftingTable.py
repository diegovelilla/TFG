from .ctgan import CTGAN
from .tvae import TVAE
from .tabddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .tabddpm.modules import MLPDiffusion
from .tabddpm.ddpm import DiffusionModel
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

from scipy.spatial.distance import mahalanobis
from scipy.stats import ks_2samp

class CraftingTable():
    """
    CraftingTable class. This class wraps different models for tabular data generation.
    These are the current models that the class holds.

    Models:
        - CTGAN: (GAN)
            Paper: https://arxiv.org/abs/1907.00503
            Code: https://github.com/sdv-dev/CTGAN
        - TVAE: (VAE)
            Paper: https://arxiv.org/abs/1907.00503
            Code: https://github.com/sdv-dev/CTGAN
    
    Attr:
        model (object): Instantiation of current selected model.
        model_hyperparameters (dict): Dictionary containing all supported models 
                                      as keys with a dictionary detailing all hyperparameters for each one.
        model_name (str): String representing a valid name of one of the supported models.
        models_available (dict): Dictionary containing all pairs (name, model) of all models available.    
        eval_models (dict): Dictionary containing all machine learning models available for `eval_ml` separated 
                                      between regression and classification tasks.
        eval_metrics (dict): Dictionary containing all evaluation metrics available for `eval_ml` separated 
                                      between regression and classification tasks.
    """
    def __init__(self):
        """
        Initializes the CraftingTable object.
        """
        self.model = None
        self.models_available = {"ctgan": CTGAN, "tvae": TVAE, "diffusion": DiffusionModel}
        self.model_hyperparameters = {
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
        self.eval_models = {
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
        self.eval_metrics = {
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
            

    def create_model(self, model_name: str, hyperparameters: dict = {}) -> None:
        """
        Creates a new instance of the desired model and loads it into the CraftingTable object.

        Args:
            model_name (str): String representing the name of the selected model.
            hyperparameters (dict): Dictionary containing information about the 
                                            hyperparameters of the model. Defaults to {}.
        Raises:
            AssertionError: If `model_name` is None.
                            If `model_name` is not supported aka not in `self.models_available`.
                            If `hyperparameters` is not a subset of the real hyperparameter set of the model.  
        """
        assert model_name is not None, "model_name needed to create a new model."
        assert model_name in self.models_available.keys(), f"model_name not supported. Current models supported: {self.models_available.keys()}"
        assert all(key in self.model_hyperparameters[model_name] for key in hyperparameters), "Invalid set of hyperparameters."

        self.model_name = model_name
        merged_hyperparameters = {**self.model_hyperparameters[model_name], **hyperparameters}
        self.model = self.models_available[model_name](**merged_hyperparameters)
        self.model.metadata["model"]["hyperparameters"] = merged_hyperparameters
        print("Model loaded successfully!")
        

    def fit(self, data: pd.DataFrame, discrete_columns: list, epochs: int = None, verbose: bool = False) -> dict:
        """
        Fit the model to the training data. Updates metadata.

        Args:
            data (pandas.DataFrame): Pandas DataFrame containing the training data.
            discrete_columns (list): List of discrete columns to be used to generate the Conditional
                                            Vector. This list should contain the column names.
            epochs (int): Number of epochs to fit the model. Defaults to None.
        Returns:
            (dict): Fit history containing information of the training split.
        Raises:
            AssertionError: If the model has not been created or loaded previously.
                            If `data` is None.
        """
        assert self.model is not None, "Can not find model to fit. Create or load a model first."
        assert data is not None, "data needed to train the model."

        if self.model_name == "diffusion":
            loss_history = self.model.train(data,
                discrete_columns, steps=epochs, verbose=verbose)    
            self.model.metadata["model"]["fit_settings"]["fit_history"].append(loss_history)
        else:
            self.model.fit(data, discrete_columns, verbose, epochs)

        return self.model.metadata["model"]["fit_settings"]["fit_history"][-1]

        
    def sample(self, num_samples: int, condition_column: str = None, condition_value: str = None, force_value: bool = False) -> pd.DataFrame:
        """
        Samples synthetic data from the fitted model.

        Args:
            num_samples (int): Number of rows to sample from the model.
            condition_column (str): **Only used if the selected model is ctgan**. 
                                        Selects column to fix value for ctgan. If a column is 
                                        not selected, the model will sample unconditionally. Defaults to None.
            condition_value (str): **Only used if the selected model is ctgan**. Selects value 
                                        to fix for ctgan. If a value is not selected, the model 
                                        will sample unconditionally. Defaults to None.
            force_value (bool): **Only used if the selected model is ctgan**. If True, all samples will contain 
                                        condition_value in their condition_column. If False, these types of rows will 
                                        only have a higher probabilty of being sampled but not be assured. 
                                        Defaults to False.
        Returns:
            (pandas.DataFrame): Pandas DataFrame containing the samples.
        
        Raises:
            AssertionError: If the model has not been created or loaded previously.
                            If `num_samples` is not a positive integer.

            Warning:    If `condition_column` or `condition_value` are passed and the selected model is not `ctgan`.
        """
        assert self.model is not None, "Can not find model to sample from. Create or load a model first."
        assert num_samples > 0, "num_samples must be a positive integer."

        if self.model_name == "ctgan":
            return self.model.sample(num_samples, condition_column, condition_value, force_value)
        elif self.model_name == "diffusion":
            return self.model.sample(num_samples=num_samples)
        else:
            if condition_column is not None or condition_value is not None:
                raise Warning("condition_column and condition_value not available for current model.")
            return self.model.sample(num_samples)

        
    def get_metadata(self) -> dict:
        """
        Returns metadata about the model being used.

        Returns:
            (dict): Dictionary containing up-to-date Metadata about the model.
        
        Raises:
            AssertionError:  If the model has not been created or loaded previously.
        """
        assert self.model is not None, "Can not find model to get metadata from. Create or load a model first."

        return self.model._get_metadata()
    
    
    def save(self, path: str = "model.pt") -> None:
        """
        Saves the model into a .pt file in the selected path.

        Args:
            path (str): Path to save the model. Defaults to "model.pt".
        
        Raises:
            AssertionError: If the model has not been created or loaded previously.
                            If `path` is not a string.
                            If `path` is not a '.pt' or '.pth' extension.
        """
        assert self.model is not None, "Can not find model to save. Create or load a model first."
        assert isinstance(path, str), "Path must be a string."
        assert path.endswith(('.pt', '.pth')), "Path must end with .pt or .pth extension."

        self.model.save(path)
        print("Model saved successfully!")


    def load(self, path: str) -> None:
        """
        Loads the model from a **.pt** file in the selected path. Method 
        *create_model()* **MUST** be called before loading the model.

        Args:
            path (str): Path to load the model from.
    
        Raises:
            AssertionError: If the model has not been created or loaded previously.
                            If `path` is not a string.
                            If `path` is not an existing file.
        """
        assert self.model is not None, "Can't load model without creating one first. Run .create_model() with a valid model_name first."
        assert isinstance(path, str), "Path must be a string."
        assert os.path.isfile(path), f"File not found at the given path: {path}. Please check the file path."
    
        model, metadata = self.model.load(path)
        self.model = model
        self.model.metadata = metadata
        print("Model loaded successfully")

    
    def eval_stat(self, real_data: pd.DataFrame, test: str, fake_data: pd.DataFrame = None, classifier: str = None):
        """
        Evaluates synthetic data quality by comparing fake samples to real data via statistical tests. 

        Args:
            real_data (pandas.DataFrame): Pandas DataFrame containing samples of real data.
            test (str): String indicating statistical test to perform.
        
        Raise: 
            ValueError: If `test` is not one of the available ones.
        """
        
        real_data_transformed = np.array(self.model._transformer.transform(real_data))
        if fake_data is None:
            n = real_data.shape[0] 
            fake_data = np.array(self.model._transformer.transform(self.sample(n)))

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
            assert classifier is not None and classifier in self.eval_models['classification'].keys(), "Classifier not selected/unavailable."
            classifier_model = self.eval_models['classification'][classifier]()
            X = np.vstack([real_data_transformed, fake_data])
            y = np.concatenate([np.ones(real_data_transformed.shape[0]), np.zeros(fake_data.shape[0])])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            classifier_model.fit(X_train, y_train)
            y_pred = classifier_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            return {"accuracy": accuracy, "report": report}
            
        else:
            raise ValueError("Invalid plot type. Choose from 'ks', 'mahalanobis'.")
        
        
    def eval_ml(self, real_data: pd.DataFrame, target_name: str, task: str, 
                ml: str, metrics: list, test_size: float = 0.3, fake_data: pd.DataFrame = None):
        """
        Evaluates synthetic data quality by training machine learning 
        models on both real and synthetic data and comparing their performance.

        Args:
            real_data (pandas.DataFrame): Pandas DataFrame containing samples of real data.
            ml (str): String representing the desired model to train.
        
        Raises:
            AssertionError: If `task` is not 'classification' or 'regression'.
                            If `ml` is not a valid machine learning model from the ones available.
                            If `metrics` contains a metric that it is not available.
        """
        assert task == "classification" or task == "regression", "Unknown task. Please choose between 'classification'/'regression'"
        assert ml in self.eval_models[task].keys(), f"Unknown machine learning model for {task}. Please select one from the following: {self.eval_models[task].keys()}."


        real_data_copy = real_data.copy()
        label_encoder = None
        if target_name in self.model.discrete_columns and not np.issubdtype(real_data_copy[target_name].dtype, np.number):
            label_encoder = LabelEncoder()
            real_data_copy[target_name] = label_encoder.fit_transform(real_data_copy[target_name])


        real_X = real_data_copy.drop(columns=[target_name])
        real_y = np.array(real_data_copy[target_name])

        categorical_columns = list(set(self.model.discrete_columns) - {target_name})
        categorical_columns = [col for col in categorical_columns if col in real_X.columns]

        real_X_transformed = np.array(pd.get_dummies(real_X, columns=categorical_columns))

        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            real_X_transformed, real_y, test_size=test_size, stratify=real_y if task == "classification" else None
        )

        if fake_data is None:
            n = real_X.shape[0]
            fake_data = self.sample(n).copy()

        if label_encoder is not None:
            fake_data[target_name] = label_encoder.transform(fake_data[target_name])

        categorical_columns_fake = [col for col in categorical_columns if col in fake_data.columns]
        X_train_fake = np.array(pd.get_dummies(fake_data.drop(columns=[target_name]), columns=categorical_columns_fake))
        y_train_fake = np.array(fake_data[target_name])

        ml_model = self.eval_models[task][ml]()
        ml_model.fit(X_train_real, y_train_real)
        y_pred_real = ml_model.predict(X_test_real)

        ml_model.fit(X_train_fake, y_train_fake)
        y_pred_fake = ml_model.predict(X_test_real)
        
        metric_results = {"real": {}, "fake": {}}
        for metric in metrics:
            if metric in self.eval_metrics[task]:
                if metric == "classification_report":
                    metric_results["real"][metric] = pd.DataFrame(self.eval_metrics[task][metric](y_test_real, y_pred_real, output_dict=True, )).T
                    metric_results["fake"][metric] = pd.DataFrame(self.eval_metrics[task][metric](y_test_real, y_pred_fake, output_dict=True)).T
                else:
                    metric_results["real"][metric] = self.eval_metrics[task][metric](y_test_real, y_pred_real)
                    metric_results["fake"][metric] = self.eval_metrics[task][metric](y_test_real, y_pred_fake)

        return metric_results


    def eval_plot(self, real_data: pd.DataFrame, plot_type: str, fake_data: pd.DataFrame = None, 
                  features: list = None, max_features: int = 5):
        """
        Plots real vs. fake data for visual comparison.

        Args:
            real_data (pandas.DataFrame): DataFrame containing real data samples.
            plot_type (str): Type of plot ('histogram', 'box', 'scatter', 'pair', 'pca', 'heatmap').
            fake_data (pandas.DataFrame, optional): DataFrame containing synthetic data samples.
            features (list, optional): List of feature names to plot. Defaults to all features.
            max_features (int, optional): Max number of features to include in pair plots. Defaults to 5.

        Raises:
            ValueError: If `plot_type` is not one of the available ones.
                        If no valid features are provided.
        """
        
        if fake_data is None:
            n = real_data.shape[0]
            fake_data = self.sample(n)

        real_data_copy = real_data.copy()
        fake_data_copy = fake_data.copy()

        all_features = real_data_copy.columns.tolist()
        if features is None:
            features = all_features
        else:
            features = [f for f in features if f in all_features]

        if len(features) == 0:
            raise ValueError("No valid features selected for plotting.")

        if plot_type == "histogram":
            fig, axes = plt.subplots(nrows=len(features), figsize=(8, 4 * len(features)))
            if len(features) == 1:
                axes = [axes]
            for ax, feature in zip(axes, features):
                sns.histplot(real_data_copy[feature], color='blue', label='Real', ax=ax, stat="density")
                sns.histplot(fake_data_copy[feature], color='red', label='Fake', ax=ax, stat="density")
                ax.set_title(f"Histogram of {feature}")
                ax.legend()
            plt.tight_layout()
            plt.show()

        elif plot_type == "box":
            features = [f for f in real_data.columns if pd.api.types.is_numeric_dtype(real_data[f])]
            fig, axes = plt.subplots(nrows=len(features), figsize=(8, 4 * len(features)))
            
            if len(features) == 1:
                axes = [axes]
                
            for ax, feature in zip(axes, features):

                combined_df = pd.DataFrame({
                    "Value": pd.concat([real_data_copy[feature], fake_data_copy[feature]], ignore_index=True),
                    "Category": ["Real"] * len(real_data_copy[feature]) + ["Fake"] * len(fake_data_copy[feature])
                })

                sns.boxplot(x="Category", y="Value", data=combined_df, ax=ax)

                ax.set_title(f"Boxplot of {feature}")
            
            plt.tight_layout()
            plt.show()

        elif plot_type == "scatter":
            if len(features) != 2:
                raise ValueError("Scatter plot requires exactly two features.")
            plt.figure(figsize=(8, 6))
            plt.scatter(real_data_copy[features[0]], real_data_copy[features[1]], alpha=0.5, label='Real', color='blue')
            plt.scatter(fake_data_copy[features[0]], fake_data_copy[features[1]], alpha=0.5, label='Fake', color='red')
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.title("Scatter Plot of Real vs Fake Data")
            plt.legend()
            plt.show()

        elif plot_type == "pair":
            selected_features = features[:max_features]
            real_data_copy["Source"] = "Real"
            fake_data_copy["Source"] = "Fake"
            combined_data = pd.concat([real_data_copy[selected_features + ["Source"]], 
                                       fake_data_copy[selected_features + ["Source"]]])
            sns.pairplot(combined_data, hue="Source", plot_kws={'alpha': 0.5})
            plt.show()

        elif plot_type == "pca":
            features = [f for f in real_data.columns if pd.api.types.is_numeric_dtype(real_data[f])]
            pca = PCA(n_components=2)
            real_pca = pca.fit_transform(real_data_copy[features])
            fake_pca = pca.transform(fake_data_copy[features])
            plt.figure(figsize=(8, 6))
            plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real', color='blue')
            plt.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.5, label='Fake', color='red')
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.title("PCA Projection of Real vs Fake Data")
            plt.legend()
            plt.show()

        elif plot_type == "heatmap":
            features = [f for f in real_data.columns if pd.api.types.is_numeric_dtype(real_data[f])]
            real_corr = real_data_copy[features].corr()
            fake_corr = fake_data_copy[features].corr()
            print(features)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.heatmap(real_corr, ax=axes[0], cmap="coolwarm", annot=True, fmt=".2f")
            axes[0].set_title("Real Data Correlation")
            sns.heatmap(fake_corr, ax=axes[1], cmap="coolwarm", annot=True, fmt=".2f")
            axes[1].set_title("Fake Data Correlation")
            plt.show()
        else:
            raise ValueError("Invalid plot type. Choose from 'histogram', 'box', 'scatter', 'pair', 'pca', 'heatmap'.")
