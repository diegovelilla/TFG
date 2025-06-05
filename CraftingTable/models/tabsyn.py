from .base import BaseModel
import os
import torch
import time
from tqdm import tqdm
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from .tabsyn_utils.vae.model import Model_VAE, Encoder_model, Decoder_model
from .tabsyn_utils.vae.utils_train import TabularDataset
from .tabsyn_utils.model import MLPDiffusion, Model
from .tabsyn_utils.diffusion_utils import sample as diffusion_sample


def compute_loss(X_num, Recon_X_num, mu_z, logvar_z):
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    
    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, loss_kld

class TabSyn(BaseModel):

    def __init__(
        self,
        dim_t: int = 512,
    ):
        super().__init__()
        self.d_in = None
        self.dim_t = dim_t
        self.metadata['model']['model_type'] = "TabSyn"
        self.metadata['model']['hyperparmeters'] = {"d_in": None, 
                                                    "dim_t": self.dim_t}
        
    def __repr__(self):
        return (f"TabSyn(d_in={self.d_in}, dim_t={self.dim_t})")
    
    def _fit_vae(
        self,
        X_num,
        num_epochs=800,
        batch_size=4096,
        lr=1e-3,
        weight_decay=0,
        d_token=4,
        token_bias=True,
        n_head=1,
        factor=32,
        num_layers=2,
        max_beta=1e-2,
        min_beta=1e-5,
        lambd=0.7,
        test_size=0.2,
        random_state=42,
        device='cuda',
        verbose = True
    ):
        # Create checkpoint directory
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(curr_dir, 'ckpt')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        model_save_path = os.path.join(ckpt_dir, 'vae_model.pt')
        encoder_save_path = os.path.join(ckpt_dir, 'encoder.pt')
        decoder_save_path = os.path.join(ckpt_dir, 'decoder.pt')

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
            num_layers,
            d_numerical,
            categories,
            d_token,
            n_head=n_head,
            factor=factor,
            bias=token_bias,
        ).to(device)

        pre_encoder = Encoder_model(
            num_layers,
            d_numerical,
            categories,
            d_token,
            n_head=n_head,
            factor=factor,
        ).to(device)

        pre_decoder = Decoder_model(
            num_layers,
            d_numerical,
            categories,
            d_token,
            n_head=n_head,
            factor=factor,
        ).to(device)

        pre_encoder.eval()
        pre_decoder.eval()

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.95, patience=10, verbose=True
        )

        best_train_loss = float('inf')
        current_lr = optimizer.param_groups[0]['lr']
        patience_counter = 0
        beta = max_beta

        start_time = time.time()
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader, total=len(train_loader))
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

            curr_loss_gauss = 0.0
            curr_loss_kl = 0.0
            curr_count = 0

            for batch_num in pbar:
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

                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    current_lr = new_lr
                    print(f"Learning rate updated: {current_lr}")

                # Save best model
                if val_loss < best_train_loss:
                    best_train_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), model_save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= 10 and beta > min_beta:
                        beta *= lambd
                        patience_counter = 0

            print(
                f"epoch: {epoch}, beta = {beta:.6f}, Train MSE: {num_loss:.6f}, "
                f"Val MSE: {val_mse_loss.item():.6f}"
            )

        end_time = time.time()
        print(f"Training time: {(end_time - start_time)/60:.4f} mins")

        # Saving latent embeddings
        with torch.no_grad():
            # pre_encoder.load_state_dict(model.encoder.state_dict())
            # pre_decoder.load_state_dict(model.decoder.state_dict())
            pre_encoder.load_weights(model)
            pre_decoder.load_weights(model)

            torch.save(pre_encoder.state_dict(), encoder_save_path)
            torch.save(pre_decoder.state_dict(), decoder_save_path)

            X_train_num = X_train_num.to(device)
            # X_train_cat = X_train_cat.to(device)

            print('Successfully loaded and saved the model!')

            train_z = pre_encoder(X_train_num, None).detach().cpu().numpy()
            np.save(os.path.join(ckpt_dir, 'train_z.npy'), train_z)
            print('Successfully saved pretrained embeddings to disk!')
            z_path = os.path.join(ckpt_dir, 'train_z.npy')
            return z_path
        

    def _fit_MLP(
        self,
        z_path,
        batch_size=4096,
        num_epochs=500,
        lr=1e-3,
        weight_decay=0,
        device='cuda',
        ckpt_path='ckpt',
        verbose=True
    ):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(curr_dir, 'ckpt')

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

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

        denoise_fn = MLPDiffusion(in_dim, 512).to(device)

        num_params = sum(p.numel() for p in denoise_fn.parameters())
        print("the number of parameters", num_params)

        model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

        model.train()

        best_loss = float('inf')
        patience = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            
            pbar = tqdm(train_loader, total=len(train_loader))
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

            batch_loss = 0.0
            len_input = 0
            for (batch,) in pbar:
                inputs = batch.float().to(device)
                loss = model(inputs)
            
                loss = loss.mean()
                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                print("hola")
                best_loss = curr_loss
                patience = 0
                torch.save(model.state_dict(), f'{ckpt_dir}/diffusion_model.pt')
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break

            if epoch % 1000 == 0:
                torch.save(model.state_dict(), f'{ckpt_dir}/diffusion_model_{epoch}.pt')

        end_time = time.time()
        print('Time: ', end_time - start_time)

    def fit(
        self,
        train_data,
        discrete_columns,
        vae_epochs=800,
        diffusion_steps=500,
        batch_size=4096,
        lr=1e-3,
        weight_decay=0,
        d_token=4,
        token_bias=True,
        n_head=1,
        factor=32,
        num_layers=2,
        max_beta=1e-2,
        min_beta=1e-5,
        lambd=0.7,
        test_size=0.2,
        random_state=42,
        device='cuda',
        verbose=True
    ):
        self.discrete_columns = discrete_columns
        self.cont_columns = list(set(train_data.columns) - set(discrete_columns))
        X_num = self._transform_data(train_data, discrete_columns)
        # with open("ckpt/transformations.pkl", "wb") as f:
        #     pickle.dump(self.transformer, f)
        self._create_table_metadata(data=train_data)
        z_path = self._fit_vae(
            X_num = X_num,
            num_epochs=vae_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            d_token=d_token,
            token_bias=token_bias,
            n_head=n_head,
            factor=factor,
            num_layers=num_layers,
            max_beta=max_beta,
            min_beta=min_beta,
            lambd=lambd,
            test_size=test_size,
            random_state=random_state,
            device=device,
            verbose=verbose
        )
        self._fit_MLP(
            z_path=z_path,
            batch_size=batch_size,
            num_epochs=diffusion_steps,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            ckpt_path='ckpt',
            verbose=verbose
        )

    def sample(
        self,
        num_samples,
        device='cuda',
    ):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = f'{curr_dir}/ckpt'
        embedding_save_path = f'{curr_dir}/ckpt/train_z.npy'
        train_z = torch.tensor(np.load(embedding_save_path)).float()

        train_z = train_z[:, 1:, :]
        B, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim
        
        train_z = train_z.view(B, in_dim)
        in_dim = train_z.shape[1] 

        mean = train_z.mean(0)

        denoise_fn = MLPDiffusion(in_dim, 512).to(device)
        
        model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

        model.load_state_dict(torch.load(f'{ckpt_path}/diffusion_model.pt'))

        '''
            Generating samples    
        '''
        start_time = time.time()

        sample_dim = in_dim

        x_next = diffusion_sample(model.denoise_fn_D, num_samples, sample_dim, device = device)
        x_next = x_next * 2 + mean.to(device)

        syn_data = x_next.float().cpu().numpy()
        # syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

        # syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        # idx_name_mapping = info['idx_name_mapping']
        # idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        # syn_df.rename(columns = idx_name_mapping, inplace=True)
        syn_df = self.transformer.inverse_transform(syn_data)
        end_time = time.time()
        print('Time:', end_time - start_time)
        return syn_df

