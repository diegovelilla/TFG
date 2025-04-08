from ctgan import CTGAN
from datetime import datetime
import numpy as np
import pandas as pd
import torch

class CTGAN(CTGAN):
    def __init__(self, *args, **kwargs):
        """
        Initializes the CTGAN model.

        Args:
            embedding_dim (int):
                Size of the random sample passed to the Generator. Defaults to 128.
            generator_dim (tuple or list of ints):
                Size of the output samples for each one of the Residuals. A Residual Layer
                will be created for each one of the values provided. Defaults to (256, 256).
            discriminator_dim (tuple or list of ints):
                Size of the output samples for each one of the Discriminator Layers. A Linear Layer
                will be created for each one of the values provided. Defaults to (256, 256).
            generator_lr (float):
                Learning rate for the generator. Defaults to 2e-4.
            generator_decay (float):
                Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
            discriminator_lr (float):
                Learning rate for the discriminator. Defaults to 2e-4.
            discriminator_decay (float):
                Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
            batch_size (int):
                Number of data samples to process in each step. Defaults to 500.
            discriminator_steps (int):
                Number of discriminator updates to do for each generator update.
                From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
                default is 5. Default used is 1 to match original CTGAN implementation.
            log_frequency (boolean):
                Whether to use log frequency of categorical levels in conditional
                sampling. Defaults to ``True``.
            verbose (boolean):
                Whether to have print statements for progress results. Defaults to ``False``.
            epochs (int):
                Number of training epochs. Defaults to 300.
            pac (int):
                Number of samples to group together when applying the discriminator.
                Defaults to 10.
            cuda (bool):
                Whether to attempt to use cuda for GPU computation.
                If this is False or CUDA is not available, CPU will be used.
                Defaults to ``True``.
        """
        self.metadata = {
            "table": {
                "random_seed": "",
                "columns": {

                },
                "correlations": {
                    
                },
                
            },
            "model":{
                "model_type": "CTGAN",
                "hyperparameters": {

                },
                "fit_settings": {
                    "times_fitted": 0,
                    "fit_history": [
                        
                    ]
                }
            }
        }
        super().__init__(*args, **kwargs)


    def fit(self, train_data, discrete_columns, verbose, epochs):
        """
        Fit the model to the training data. Updates metadata.

        Args:
            train_data (pandas.DataFrame): Training Data.
            discrete_columns (list): List of discrete columns to be used to generate the Conditional
                                     Vector. This list should contain the column names.
            verbose (bool): If True, displays the time remaining and the current loss.
            epochs (int): The number of epochs to train the model.
        """
        self._verbose = verbose
        self.discrete_columns = discrete_columns
        self.continous_columns = list(set(train_data.columns) - set(discrete_columns))

        if self.metadata["table"]["columns"] == {}:
            for column in train_data.columns:
                if column in self.continous_columns:
                    """
                    Dont get null data metadata bc will be treated inside fit function.
                    """
                    column_dict = {
                        "dtype": str(train_data[column].dtype),
                        "max": np.max(train_data[column]).item(),
                        "min": np.min(train_data[column]).item(),
                        "avg": np.average(train_data[column]).item(),
                        "std": np.std(train_data[column]).item(),
                        "median": np.median(train_data[column]).item(),
                    }
                else:
                    column_dict = {
                        "dtype": str(train_data[column].dtype),
                        "mode": train_data[column].mode().iloc[0],
                        "nunique": train_data[column].nunique(),
                        "value_counts": train_data[column].value_counts().to_dict(),
                    }
                self.metadata["table"]["columns"][column] = column_dict

        if self.metadata["table"]["correlations"] == {}:
            self.metadata["table"]["correlations"] = train_data[self.continous_columns].corr().to_dict()

        pre_time = datetime.now()
        self._epochs = epochs
        super().fit(train_data, discrete_columns=discrete_columns, epochs=epochs)
        post_time = datetime.now()

        fit_duration = post_time - pre_time

        fit_dict = {
                "Time_of_fit": pre_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Fit_duration": str(fit_duration).split('.')[0],
                "Loss": self.loss_values
        }

        self.metadata["model"]["fit_settings"]["times_fitted"] += 1
        self.metadata["model"]["fit_settings"]["fit_history"].append(fit_dict)

    def sample(self, num_samples, condition_column, condition_value, force_value):
        if force_value:
            collected_samples = []
            while len(collected_samples) < num_samples:
                # Sample a batch from the model
                batch = super().sample(num_samples, condition_column, condition_value)
                valid_rows = batch[batch[condition_column] == condition_value]
                collected_samples.extend(valid_rows.to_dict(orient="records"))
            return pd.DataFrame(collected_samples[:20])
        else:
            return super().sample(num_samples, condition_column, condition_value)

    def _get_metadata(self):
        """
        Returns metadata of the model.

        Returns:
            (dict): Dictionary containing up-to-date Metadata about the model.
        """
        return self.metadata
    
    def save(self, path):
        """
        Saves model information and metadata to selected path.

        """
        model_data = {
            "model": self,
            "metadata": self.metadata
        }
        torch.save(model_data, path)

    def load(self, path):
        """
        Loads a model from a given path.

        Returns:
            model (dict): Dictionary containing the model information.
            metadata (dict): Dictionary containing the metadata information.
        """
        model_data = torch.load(path, weights_only=False)
        model = model_data['model']
        metadata = model_data['metadata']
        return model, metadata