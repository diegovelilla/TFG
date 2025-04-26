import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from ctgan.data_transformer import DataTransformer
from .tabddpm.utils_train import get_model
from .tabddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .tabddpm.train import Trainer
from .base import BaseModel
from datetime import datetime
    
class TabDDPM_MLP(BaseModel):
    
    def __init__(
        self, 
        dim_t=256, 
        d_layers=(8, 16), 
        dropout=0.1, 
    ):
        super().__init__()
        self.d_in = None
        self.d_out = None
        self.dim_t = dim_t
        self.d_layers = d_layers
        self.dropout = dropout
        self.rtdl_params = {
            "d_layers": d_layers,
            "dropout": dropout,
        }
        self.metadata['model']['model_type'] = "TabDDPM_MLP"
        self.metadata['model']['hyperparmeters'] = {"d_in": None, 
                                                    "d_out": None, 
                                                    "dim_t": dim_t,
                                                    "d_layers": d_layers,
                                                    "dropout": dropout}

    def __repr__(self):
        return (f"TabDDPM_MLP(d_in={self.d_in}, d_out={self.d_out}, dim_t={self.dim_t}, "
            f"d_layers={self.d_layers}, dropout={self.dropout})")
    
    def _make_dataset(
        self,
        data_input,
        discrete_columns: list = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42
    ):
        transformer = DataTransformer()
        transformer.fit(data_input, discrete_columns)
        self.transformer = transformer

        data_transformed = transformer.transform(data_input)

        train_df, temp_df = train_test_split(
            data_transformed, test_size=(test_size + val_size), random_state=seed
        )
        val_ratio = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(temp_df, test_size=val_ratio, random_state=seed)

        train_tensor = torch.from_numpy(train_df.astype("float32"))
        val_tensor = torch.from_numpy(val_df.astype("float32"))
        test_tensor = torch.from_numpy(test_df.astype("float32"))

        train_dataset = TensorDataset(train_tensor, train_tensor)
        val_dataset = TensorDataset(val_tensor, val_tensor)
        test_dataset = TensorDataset(test_tensor, test_tensor)

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "transformer": transformer
        }
    
    def fit(
        self, 
        train_data, 
        discrete_columns, 
        steps=1000,
        lr=0.005,
        weight_decay=1e-4,
        batch_size=2048,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        device='cuda',
        verbose=True
    ):
        dataset_dict = self._make_dataset(
            train_data,
            discrete_columns
        )

        self.d_in = dataset_dict['train'][0][0].shape[0]
        self.d_out = self.d_in
        self.metadata['model']['hyperparmeters'] = {"d_in": self.d_in, 
                                                    "d_out": self.d_out}
        self.rtdl_params['d_in'] = self.d_in
        self.rtdl_params['d_out'] = self.d_out
        model_params = {
            "rtdl_params": self.rtdl_params
        }
        self.model = get_model(model_name="mlp", model_params=model_params)
        self.model.to(device)

        self.column_names = list(train_data.columns)
        self.discrete_columns = discrete_columns
        self.cont_columns = list(set(train_data.columns) - set(discrete_columns))

        if self.metadata["table"]["columns"] == {}:
            for column in train_data.columns:
                if column in self.cont_columns:
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
            self.metadata["table"]["correlations"] = train_data[self.cont_columns].corr().to_dict()

        self.num_trans_features = dataset_dict['train'][0][0].shape[0]
        self.transformation = dataset_dict['transformer']
        train_dataset = dataset_dict["train"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        diffusion = GaussianMultinomialDiffusion(
            num_classes=np.array([0]),
            num_numerical_features=self.num_trans_features,
            denoise_fn=self.model,
            gaussian_loss_type=gaussian_loss_type,
            num_timesteps=num_timesteps,
            scheduler=scheduler,
            device=device
        )
        diffusion.to(device)
        diffusion.train()
        trainer = Trainer(diffusion, train_loader, lr, weight_decay, steps, device)
        pre_time = datetime.now()
        loss_history = trainer.run_loop(verbose)
        post_time = datetime.now()
        loss_history.pop("mloss")
        loss_history.pop("gloss")
        loss_history['Loss'] = loss_history.pop("loss")
        loss_history['Step'] = loss_history.pop("step")
        fit_duration = post_time - pre_time
        fit_dict = {
                "Time_of_fit": pre_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Fit_duration": str(fit_duration).split('.')[0],
                "Loss": loss_history
        }
        self.metadata["model"]["fit_settings"]["times_fitted"] += 1
        self.metadata["model"]["fit_settings"]["fit_history"].append(fit_dict)

    def sample(
        self,
        num_samples,
        batch_size=2000,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        device='cuda',
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