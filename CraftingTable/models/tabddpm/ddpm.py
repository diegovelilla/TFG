import torch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from ctgan.data_transformer import DataTransformer
from .utils_train import get_model
from .gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .train import Trainer
from datetime import datetime

class DiffusionModel:
    def __init__(self, d_layers=[128], device=torch.device('cuda:0'), seed=0):
        self.seed = seed
        self.d_layers = d_layers
        self.device = device
        self.metadata = {
            "table": {
                "random_seed": "",
                "columns": {

                },
                "correlations": {
                    
                },
                
            },
            "model":{
                "model_type": "diffusion",
                "hyperparameters": {

                },
                "fit_settings": {
                    "times_fitted": 0,
                    "fit_history": [
                        
                    ]
                }
            }
        }

    def make_dataset(
        self,
        data_input,
        discrete_columns: list = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42
    ):
        train_df, temp_df = train_test_split(
            data_input, test_size=(test_size + val_size), random_state=seed
        )
        val_ratio = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(temp_df, test_size=val_ratio, random_state=seed)

        transformer = DataTransformer()
        transformer.fit(train_df,discrete_columns)
        self._transformer = transformer
        train_transformed = transformer.transform(train_df)
        val_transformed = transformer.transform(val_df)
        test_transformed = transformer.transform(test_df)

        train_tensor = torch.from_numpy(train_transformed.astype("float32"))
        val_tensor = torch.from_numpy(val_transformed.astype("float32"))
        test_tensor = torch.from_numpy(test_transformed.astype("float32"))

        train_dataset = TensorDataset(train_tensor, train_tensor)
        val_dataset = TensorDataset(val_tensor, val_tensor)
        test_dataset = TensorDataset(test_tensor, test_tensor)

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "transformer": transformer
        }

        
    
    def train(
        self,
        data,
        discrete_columns,
        steps=1000,
        lr=0.005,
        weight_decay=1e-4,
        batch_size=2048,
        dropout = 0.0,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        device=torch.device('cuda:0'),
        verbose=True
    ):
        self.steps = steps
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type

        self.column_names = list(data.columns)
        self.discrete_columns = discrete_columns
        self.continous_columns = list(set(data.columns) - set(discrete_columns))
        
        model_params = {"rtdl_params":{}}

        if self.metadata["table"]["columns"] == {}:
            for column in data.columns:
                if column in self.continous_columns:
                    """
                    Dont get null data metadata bc will be treated inside fit function.
                    """
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
            self.metadata["table"]["correlations"] = data[self.continous_columns].corr().to_dict()


        dataset_dict = self.make_dataset(
            data,
            discrete_columns
        )
        
        
        self.num_trans_features = dataset_dict['train'][0][0].shape[0]
        self.transformation = dataset_dict['transformer']
        train_dataset = dataset_dict["train"]

        sample_feature, _ = train_dataset[0]
        d_in = sample_feature.shape[0]
        model_params["rtdl_params"]["d_in"] = d_in
        model_params["rtdl_params"]["d_out"] = d_in
        model_params["rtdl_params"]["d_layers"] = self.d_layers
        model_params["rtdl_params"]["dropout"] = dropout
        
        self.model = get_model('mlp', model_params)
        self.model.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        diffusion = GaussianMultinomialDiffusion(
            num_classes=np.array([0]),
            num_numerical_features=d_in,
            denoise_fn=self.model,
            gaussian_loss_type=gaussian_loss_type,
            num_timesteps=num_timesteps,
            scheduler=scheduler,
            device=device
        )
        diffusion.to(device)
        diffusion.train()
        #print("Starting Training...")
        trainer = Trainer(diffusion, train_loader, lr, weight_decay, steps, device)

        

        pre_time = datetime.now()
        loss_history = trainer.run_loop(verbose)
        post_time = datetime.now()
        fit_duration = post_time - pre_time
        fit_dict = {
                "Time_of_fit": pre_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Fit_duration": str(fit_duration).split('.')[0],
                "Loss": loss_history
        }
        self.metadata["model"]["fit_settings"]["times_fitted"] += 1
        self.metadata["model"]["fit_settings"]["fit_history"].append(fit_dict)
        return loss_history

        # Optionally, save losses and model weights
        # trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
        # torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
        # torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
   
   
    def sample( 
            self,                              
            num_samples,               
            batch_size=2000,
            num_timesteps=1000,
            gaussian_loss_type='mse',
            scheduler='cosine',
            device=torch.device('cuda'),
            ddim=True
        ):
        self.model.to(device)
        diffusion = GaussianMultinomialDiffusion(
            num_classes=np.array([0]),
            num_numerical_features=self.num_trans_features,
            denoise_fn=self.model,
            num_timesteps=num_timesteps,
            gaussian_loss_type=gaussian_loss_type,
            scheduler=scheduler,
            device=device
        )
        diffusion.to(device)
        diffusion.eval()
        x_gen, _ = diffusion.sample_all(num_samples, batch_size, ddim=ddim)
        X_gen = x_gen.cpu().numpy()
        X_num_trans = self.transformation.inverse_transform(X_gen)
        df_features = pd.DataFrame(X_num_trans, columns=self.column_names)
        
        return df_features

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