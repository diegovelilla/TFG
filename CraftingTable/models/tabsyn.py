from .base import BaseModel
import os
import torch
import time
from tqdm import tqdm
import pickle
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from .tabsyn_utils.vae.model import Model_VAE, Encoder_model, Decoder_model
from .tabsyn_utils.vae.utils_train import TabularDataset
from .tabsyn_utils.model import MLPDiffusion, Model
from .tabsyn_utils.diffusion_utils import sample as diffusion_sample
import pandas as pd
from torch.cuda import is_available


def compute_loss(X_num, Recon_X_num, mu_z, logvar_z):
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    
    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, loss_kld

class TabSyn(BaseModel):

    def __init__(
        self,
        n_head_vae: int = 1,
        d_token_vae: int = 4,
        factor_vae: int = 32,
        num_layers_vae: int = 2,
        dim_t_mlp: int = 512,

    ):
        super().__init__()
        self.n_head = n_head_vae
        self.d_token = d_token_vae
        self.factor = factor_vae
        self.num_layers = num_layers_vae
        self.dim_t = dim_t_mlp
        self.metadata['model']['model_type'] = "TabSyn"
        self.metadata['model']['hyperparameters'] = {"num_layers_vae": num_layers_vae,
                                                    "factor_vae": factor_vae,
                                                    "n_head_vae": n_head_vae,
                                                    "d_token_vae": d_token_vae,
                                                    "dim_t_mlp": dim_t_mlp,}
        
    def __repr__(self):
        return (f"TabSyn(num_layers_vae={self.num_layers}, factor_vae={self.factor}, n_head_vae={self.n_head}, d_token_vae={self.d_token}, dim_t_mlp={self.dim_t})")
    
    def _fit_vae(
        self,
        X_num,
        num_epochs=800,
        batch_size=4096,
        lr=1e-3,
        weight_decay=0,
        token_bias=True,
        max_beta=1e-2,
        min_beta=1e-5,
        lambd=0.7,
        test_size=0.2,
        random_state=42,
        device='cuda',
        verbose = False,
        save_final_model=False,
        save_folder='saves'
    ):
        if is_available() and device == 'cuda':
            device = 'cuda'
            if verbose:
                print("Using CUDA for training.")
        else:
            device = 'cpu'

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        model_save_path = os.path.join(save_folder, 'vae_model.pt')
        encoder_save_path = os.path.join(save_folder, 'encoder.pt')
        decoder_save_path = os.path.join(save_folder, 'decoder.pt')

        # Transform data using the provided transformer
        
        # X_cat = None
        categories = []
        d_numerical = X_num.shape[1]
        # Split into train and test sets
        X_num_train, X_num_test = train_test_split(
            X_num, test_size=test_size, random_state=random_state
        )
        # X_cat_train = X_cat_test = None

        X_train_num = torch.tensor(X_num_train).float()
        # X_train_cat = torch.tensor(X_cat_train)
        X_test_num = torch.tensor(X_num_test).float().to(device)
        # X_test_cat = torch.tensor(X_cat_test).to(device)

        # Create DataLoader
        train_dataset = TabularDataset(X_train_num)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )

        # Instantiate models
        model = Model_VAE(
            self.num_layers,
            d_numerical,
            categories,
            self.d_token,
            n_head=self.n_head,
            factor=self.factor,
            bias=token_bias,
        ).to(device)

        pre_encoder = Encoder_model(
            self.num_layers,
            d_numerical,
            categories,
            self.d_token,
            n_head=self.n_head,
            factor=self.factor,
        ).to(device)

        pre_decoder = Decoder_model(
            self.num_layers,
            d_numerical,
            categories,
            self.d_token,
            n_head=self.n_head,
            factor=self.factor,
        ).to(device)

        pre_encoder.eval()
        pre_decoder.eval()

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.95, patience=10,
        )

        best_train_loss = float('inf')
        current_lr = optimizer.param_groups[0]['lr']
        patience_counter = 0
        beta = max_beta

        loss_records = []
        pbar = tqdm(range(num_epochs), disable=(not verbose))
        for epoch in pbar:

            curr_loss_gauss = 0.0
            curr_loss_kl = 0.0
            curr_count = 0

            for batch_num in train_loader:
                model.train()
                optimizer.zero_grad()

                batch_num = batch_num.to(device)
                # batch_cat = batch_cat.to(device)

                Recon_X_num, _, mu_z, std_z = model(batch_num, None)

                loss_mse, loss_kld = compute_loss(
                    batch_num, Recon_X_num, mu_z, std_z
                )

                loss = loss_mse + beta * loss_kld
                loss.backward()
                pbar.set_description(f"VAE Loss: {loss.item():.4f}")
                optimizer.step()

                batch_length = batch_num.shape[0]
                curr_count += batch_length
                curr_loss_gauss += loss_mse.item() * batch_length
                curr_loss_kl += loss_kld.item() * batch_length

            num_loss = curr_loss_gauss / curr_count
            kl_loss = curr_loss_kl / curr_count

            # Evaluation on test set
            model.eval()
            with torch.no_grad():
                Recon_X_num, _, mu_z, std_z = model(X_test_num, None)
                val_mse_loss, val_kl_loss = compute_loss(
                    X_test_num, Recon_X_num, mu_z, std_z
                )
                val_loss = val_mse_loss.item()  # mse term weighted by 0 as before
                loss_records.append(val_loss)
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    current_lr = new_lr
                    if verbose:
                        print(f"Learning rate updated: {current_lr}")

                # Save best model
                if save_final_model and val_loss < best_train_loss:
                    best_train_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), model_save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= 10 and beta > min_beta:
                        beta *= lambd
                        patience_counter = 0

        # Saving latent embeddings
        with torch.no_grad():
            # pre_encoder.load_state_dict(model.encoder.state_dict())
            # pre_decoder.load_state_dict(model.decoder.state_dict())
            pre_encoder.load_weights(model)
            pre_decoder.load_weights(model)

            if save_final_model:
                torch.save(pre_encoder.state_dict(), encoder_save_path)
                torch.save(pre_decoder.state_dict(), decoder_save_path)

            X_train_num = X_train_num.to(device)
            # X_train_cat = X_train_cat.to(device)

            train_z = pre_encoder(X_train_num, None).detach().cpu().numpy()
            np.save(os.path.join(save_folder, 'train_z.npy'), train_z)
            if verbose:
                print('Successfully saved pretrained embeddings to disk!')
            z_path = os.path.join(save_folder, 'train_z.npy')
            train_z = torch.tensor(train_z).float()
            train_z = train_z[:, 1:, :]
            B, num_tokens, token_dim = train_z.size()
            in_dim = num_tokens * token_dim
            train_z = train_z.view(B, in_dim)
            self.in_dim = train_z.shape[1] 
            self.mean = train_z.mean(0)
            loss_df = pd.DataFrame({
                'Epoch': list(range(0, len(loss_records))),
                'Vae Loss': loss_records
            })
            return z_path, loss_df
        

    def _fit_MLP(
        self,
        z_path,
        batch_size=4096,
        num_epochs=500,
        weight_decay=0,
        lr=1e-3,
        device='cuda',
        verbose=False,
        save_final_model=False,
        save_folder='saves'
    ):
        if is_available() and device == 'cuda':
            device = 'cuda'
            if verbose:
                print("Using CUDA for training.")
        else:
            device = 'cpu'
            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        train_z = torch.tensor(np.load(z_path)).float()
        train_z = train_z[:, 1:, :]
        B, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim
        
        train_z = train_z.view(B, in_dim)

        in_dim = train_z.shape[1] 

        mean, std = train_z.mean(0), train_z.std(0)

        train_z = (train_z - mean) / 2
        train_data = TensorDataset(train_z)

        train_loader = DataLoader(
            train_data,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 4,
        )

        self.denoise_fn = MLPDiffusion(in_dim, self.dim_t).to(device)

        self.model = Model(denoise_fn = self.denoise_fn, hid_dim = train_z.shape[1]).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

        self.model.train()

        loss_records = []
        best_loss = float('inf')
        patience = 0
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            
            batch_loss = 0.0
            len_input = 0
            for (batch,) in train_loader:
                inputs = batch.float().to(device)
                loss = self.model(inputs)
            
                loss = loss.mean()
                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(f"MLP Loss: {loss.item():.4f}")

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)
            loss_records.append(curr_loss)
            if save_final_model and curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(self.model.state_dict(), f'{save_folder}/diffusion_model.pt')
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break

            if epoch % 1000 == 0 and save_final_model:
                torch.save(self.model.state_dict(), f'{save_folder}/diffusion_model_{epoch}.pt')

        loss_df = pd.DataFrame({
                'Epoch': list(range(0, len(loss_records))),
                'MLP Loss': loss_records
            })
        return loss_df

    def fit(
        self,
        train_data: pd.DataFrame,
        discrete_columns: list[str],
        vae_epochs: int = 800,
        mlp_epochs: int = 500,
        batch_size: int = 4096,
        weight_decay: float = 0,
        lr: float = 1e-3,
        token_bias: bool = True,
        max_beta: float = 1e-2,
        min_beta: float = 1e-5,
        lambd: float = 0.7,
        test_size: float = 0.2,
        random_state: int = 42,
        device: str = 'cuda',
        verbose: bool = False,
        save_final_model: bool = False,
        save_folder: str = 'saves'
    ):
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("train_data must bea a pandas DataFrame.")
        if not isinstance(discrete_columns, list) and all(isinstance(col, str) and col in train_data.columns for col in discrete_columns):
            raise TypeError("discrete_columns must be a list of column names.")
        
        if is_available() and device == 'cuda':
            device = 'cuda'
            if verbose:
                print("Using CUDA for training.")
        else:
            device = 'cpu'
            if verbose:
                print("Using CPU for training.")
                
        self.discrete_columns = discrete_columns
        self.cont_columns = list(set(train_data.columns) - set(discrete_columns))
        X_num = self._transform_data(train_data, discrete_columns)
        # with open("ckpt/transformations.pkl", "wb") as f:
        #     pickle.dump(self.transformer, f)
        self._create_table_metadata(data=train_data)
        pre_time_vae = datetime.now()
        z_path, vae_loss = self._fit_vae(
            X_num = X_num,
            num_epochs=vae_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            token_bias=token_bias,
            max_beta=max_beta,
            min_beta=min_beta,
            lambd=lambd,
            test_size=test_size,
            random_state=random_state,
            device=device,
            verbose=verbose,
            save_final_model=save_final_model,
            save_folder=save_folder
        )
        post_time_vae = datetime.now()
        fit_time_vae = post_time_vae - pre_time_vae

        pre_time_mlp = datetime.now()
        mlp_loss = self._fit_MLP(
            z_path=z_path,
            batch_size=batch_size,
            num_epochs=mlp_epochs,
            weight_decay=weight_decay,
            lr=lr,
            device=device,
            verbose=verbose,
            save_final_model=save_final_model,
            save_folder=save_folder
        )
        post_time_mlp = datetime.now()
        fit_time_mlp = post_time_mlp - pre_time_mlp
        fit_dict = {
            "time_of_fit": pre_time_vae.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": {
                    "vae_duration": str(fit_time_vae).split('.')[0], 
                    "mlp_duration": str(fit_time_mlp).split('.')[0]
                },

            "parameters": {
                "device": device,
                "vae_epochs": vae_epochs,
                "mlp_epochs": mlp_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_beta": max_beta,
                "min_beta": min_beta,
                "lambda": lambd,
                "weight_decay": weight_decay},

            "loss": {
                "vae_loss": vae_loss,
                "mlp_loss": mlp_loss
            }
        }

        self.metadata["model"]["fit_settings"]["times_fitted"] += 1
        self.metadata["model"]["fit_settings"]["fit_history"].append(fit_dict)

    def sample(
        self,
        num_samples: int,
        device: str = 'cuda',
    ):
        if self.metadata['model']['fit_settings']['times_fitted'] == 0:
            raise RuntimeError("Model has not been fitted yet. Please fit the model before sampling.")
        '''
            Generating samples    
        '''
        if is_available() and device =='cuda':
            device = 'cuda'
        else:
            device = 'cpu'

        sample_dim = self.in_dim

        x_next = diffusion_sample(self.model.denoise_fn_D, num_samples, sample_dim, device = device)
        x_next = x_next * 2 + self.mean.to(device)

        syn_data = x_next.float().cpu().numpy()
        # syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

        # syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        # idx_name_mapping = info['idx_name_mapping']
        # idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        # syn_df.rename(columns = idx_name_mapping, inplace=True)
        syn_df = self.transformer.inverse_transform(syn_data)
        return syn_df