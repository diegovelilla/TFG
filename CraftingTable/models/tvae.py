from ctgan import TVAE as parentTVAE
from datetime import datetime
import numpy as np
import torch
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.tvae import Encoder, Decoder, _loss_function
from .base import BaseModel


class TVAE(parentTVAE, BaseModel):
    def __init__(self, embedding_dim=128, compress_dims=(128, 128), decompress_dims=(128, 128)):
        BaseModel.__init__(self)
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.metadata['model']['model_type'] = "TVAE"
        self.metadata['model']['hyperparmeters'] = {"embeddig_dim": embedding_dim, 
                                                    "compress_dims": compress_dims, 
                                                    "decompress_dims": decompress_dims}
        
    def __repr__(self):
        return f"TVAE(embedding_dim={self.embedding_dim}, compress_dims={self.compress_dims}, decompress_dims={self.decompress_dims})"

    
    def fit(self, train_data, discrete_columns, l2scale=1e-5, batch_size=500, 
            epochs=300, loss_factor=2, cuda=True, verbose=True):
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_factor = loss_factor
        self.verbose = verbose
        self.discrete_columns = discrete_columns
        self.cont_columns = list(set(train_data.columns) - set(discrete_columns))
        
        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self._device = self.device

        self._create_table_metadata(data=train_data)

        pre_time = datetime.now()
        self.epochs = epochs

        train_data = self._transform_data(data=train_data, discrete_columns=discrete_columns)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self.device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Loss: {loss:.3f}'
            iterator.set_description(iterator_description.format(loss=0))

        for i in iterator:
            loss_values = []
            batch = []
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self.device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    self.transformer.output_info_list,
                    self.loss_factor,
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

            batch.append(id_)
            loss_values.append(loss.detach().cpu().item())

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i] * len(batch),
                'Loss': loss_values,
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self.verbose:
                iterator.set_description(
                    iterator_description.format(loss=loss.detach().cpu().item())
                )

        post_time = datetime.now()

        fit_duration = post_time - pre_time

        fit_dict = {
                "time_of_fit": pre_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": str(fit_duration).split('.')[0],
                "hyperparameters": {"batch_size": batch_size,
                                    "epochs": epochs,
                                    "device": self.device,
                                    "l2_scale": l2scale,
                                    "loss_factor": loss_factor},
                "loss": self.loss_values
        }

        self.metadata["model"]["fit_settings"]["times_fitted"] += 1
        self.metadata["model"]["fit_settings"]["fit_history"].append(fit_dict)

    def sample(self, num_saples):
        return parentTVAE.sample(self, num_saples)
    
    