from ctgan import CTGAN as parentCTGAN
from .base import BaseModel
from datetime import datetime
import numpy as np
import pandas as pd

class CTGAN(parentCTGAN, BaseModel):
    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256)):
        BaseModel.__init__(self)
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.metadata['model']['model_type'] = "CTGAN"
        self.metadata['model']['hyperparmeters'] = {"embeddig_dim": embedding_dim, 
                                                    "generator_dim": generator_dim, 
                                                    "discriminator_dim": discriminator_dim}
        
    def __repr__(self):
        return f"CTGAN(embedding_dim={self.embedding_dim}, generator_dim={self.generator_dim}, discriminator_dim={self.discriminator_dim})"

    def fit(self, train_data, discrete_columns, generator_lr=2e-4, generator_decay=1e-6, 
            discriminator_lr=2e-4, discriminator_decay=1e-6, batch_size=500, discriminator_steps=1, 
            log_frequency=True, verbose=True, epochs=300, pac=10, cuda=True):
        
        self.discrete_columns = discrete_columns
        parentCTGAN.__init__(self, self.embedding_dim, self.generator_dim, self.discriminator_dim, generator_lr,
                         generator_decay, discriminator_lr, discriminator_decay, 
                         batch_size, discriminator_steps, log_frequency, verbose, 
                         epochs, pac, cuda)
        
        cont_columns = list(set(train_data.columns) - set(discrete_columns))

        if self.metadata["table"]["columns"] == {}:
            for column in train_data.columns:
                if column in cont_columns:
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
            self.metadata["table"]["correlations"] = train_data[cont_columns].corr().to_dict()

        pre_time = datetime.now()
        parentCTGAN.fit(self, train_data, discrete_columns=discrete_columns, epochs=epochs)
        post_time = datetime.now()

        fit_duration = post_time - pre_time

        fit_dict = {
                "Time_of_fit": pre_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Fit_duration": str(fit_duration).split('.')[0],
                "Loss": self.loss_values
        }

        self.metadata["model"]["fit_settings"]["times_fitted"] += 1
        self.metadata["model"]["fit_settings"]["fit_history"].append(fit_dict)

    def sample(self, num_samples, condition_column=None, condition_value=None, force_value=False):
        if force_value and condition_value != None:
            collected_samples = []
            while len(collected_samples) < num_samples:
                # Sample a batch from the model
                batch = parentCTGAN.sample(self, num_samples, condition_column, condition_value)
                valid_rows = batch[batch[condition_column] == condition_value]
                collected_samples.extend(valid_rows.to_dict(orient="records"))
            return pd.DataFrame(collected_samples[:num_samples])
        else:
            return parentCTGAN.sample(self, num_samples, condition_column, condition_value)