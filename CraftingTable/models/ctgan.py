from ctgan import CTGAN as parentCTGAN
from .base import BaseModel
from datetime import datetime
import pandas as pd
from typing import Any
import os
from torch.cuda import is_available

MAX_SAMPLING_ATTEMPTS = 10

class CTGAN(parentCTGAN, BaseModel):
    def __init__(self, embedding_dim: int = 128, generator_dim: tuple = (256, 256), discriminator_dim: tuple = (256, 256)):
        BaseModel.__init__(self)
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.metadata['model']['model_type'] = "CTGAN"
        self.metadata['model']['hyperparameters'] = {"embeddig_dim": embedding_dim, 
                                                    "generator_dim": generator_dim, 
                                                    "discriminator_dim": discriminator_dim}
        
    def __repr__(self):
        return f"CTGAN(embedding_dim={self.embedding_dim}, generator_dim={self.generator_dim}, discriminator_dim={self.discriminator_dim})"

    def fit(self, 
            train_data: pd.DataFrame, 
            discrete_columns: list[str], 
            generator_lr: float = 2e-4, 
            generator_decay: float = 1e-6, 
            discriminator_lr: float = 2e-4, 
            discriminator_decay: float = 1e-6, 
            batch_size: int = 500, 
            discriminator_steps: int = 1, 
            log_frequency: bool = True, 
            verbose: bool = False, 
            epochs: int = 300, 
            pac: int = 10, 
            device: str = 'cuda', 
            save_final_model: bool = False, 
            save_folder: str = 'saves'
            ):
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("train_data must bea a pandas DataFrame.")
        if not isinstance(discrete_columns, list) and all(isinstance(col, str) and col in train_data.columns for col in discrete_columns):
            raise TypeError("discrete_columns must be a list of column names.")
        if batch_size % pac != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by pac ({pac}).")
        if is_available() and device == 'cuda':
            cuda = True
            if verbose:
                print("Using CUDA for training.")
        else:
            cuda = False
            if verbose:
                print("Using CPU for training.")

        self.discrete_columns = discrete_columns
        parentCTGAN.__init__(self, self.embedding_dim, self.generator_dim, self.discriminator_dim, generator_lr,
                         generator_decay, discriminator_lr, discriminator_decay, 
                         batch_size, discriminator_steps, log_frequency, verbose, 
                         epochs, pac, cuda)
        
        self.cont_columns = list(set(train_data.columns) - set(discrete_columns))

        self._create_table_metadata(data=train_data)

        pre_time = datetime.now()
        parentCTGAN.fit(self, train_data, discrete_columns=discrete_columns, epochs=epochs)
        post_time = datetime.now()

        fit_duration = post_time - pre_time
        fit_dict = {
                "time_of_fit": pre_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": str(fit_duration).split('.')[0],
                "parameters": {"device": device,
                                    "epochs": epochs,
                                    "batch_size": batch_size,
                                    "generator_lr": generator_lr,
                                    "generator_decay": generator_decay,
                                    "discriminator_lr": discriminator_lr,
                                    "discriminator_decay": discriminator_decay,
                                    "discriminator_steps": discriminator_steps},
                "loss": self.loss_values
        }

        self.metadata["model"]["fit_settings"]["times_fitted"] += 1
        self.metadata["model"]["fit_settings"]["fit_history"].append(fit_dict)
        if save_final_model:
            self.save(os.path.join(save_folder, 'tvae.pt'))

    def sample(self, 
               num_samples: int , 
               condition_column: str = None, 
               condition_value: Any = None, 
               force_value: bool = False
               ):
        if self.metadata['model']['fit_settings']['times_fitted'] == 0:
            raise RuntimeError("Model has not been fitted yet. Please fit the model before sampling.")
        if condition_column is not None and condition_column not in self.metadata['table']['columns']:
            raise ValueError(f"Condition column '{condition_column}' not found in the data seen in training.")
        missed_tries = 0
        if force_value and condition_value != None:
            collected_samples = []
            while len(collected_samples) < num_samples:
                # Sample a batch from the model
                batch = parentCTGAN.sample(self, num_samples, condition_column, condition_value)
                valid_rows = batch[batch[condition_column] == condition_value]
                if valid_rows.empty:
                    missed_tries += 1
                    if missed_tries >= MAX_SAMPLING_ATTEMPTS:
                        raise RuntimeError(f"Failed to sample {num_samples} rows with condition {condition_column}={condition_value}.")
                collected_samples.extend(valid_rows.to_dict(orient="records"))
            return pd.DataFrame(collected_samples[:num_samples])
        else:
            return parentCTGAN.sample(self, num_samples, condition_column, condition_value)