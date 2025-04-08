from ctgan import TVAE
from datetime import datetime
import numpy as np
import torch
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.tvae import Encoder, Decoder, _loss_function


class TVAE(TVAE):
    def __init__(self, *args, **kwargs):
        """
    	Initializes the TVAE model.

    	Args:
    		embedding_dim (int):
        		Size of the random sample passed to the Generator. This parameter controls the dimensionality of the latent space in the model. Defaults to 128.
    		compress_dims (tuple or list of ints):
        		The dimensions of the encoder layers in the model. A Residual Layer will be created for each value provided. Defaults to (128, 128).
    		decompress_dims (tuple or list of ints):
        		The dimensions of the decoder layers in the model. A Residual Layer will be created for each value provided. Defaults to (128, 128).
    		l2scale (float):
        		Weight for L2 regularization to prevent overfitting by penalizing large weights. Defaults to 1e-5.
    		batch_size (int):
        		The number of data samples to process in each training step. A larger batch size can improve parallelization but may require more memory. Defaults to 500.
    		epochs (int):
        		The number of training epochs, or complete passes through the training dataset. Defaults to 300.
    		loss_factor (float):
        		A scaling factor for the loss function, which can be used to adjust the relative weight of different loss components in the model. Defaults to 2.
    		cuda (bool):
        		Whether to attempt to use CUDA for GPU computation. If set to False or CUDA is not available, the model will use CPU. Defaults to True.
    		verbose (bool):
        		Whether to print detailed progress messages during training. Set to True for more detailed output. Defaults to False.
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
            train_data (pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list):
                List of discrete columns to be used to generate the Conditional
                Vector. This list should contain the column names.
        """
        self.verbose = verbose
        self.discrete_columns = discrete_columns
        self.continous_columns = list(set(train_data.columns) - set(discrete_columns))

        if self.metadata["table"]["columns"] == {}:
            for column in train_data.columns:
                if column in self.continous_columns:
                    """
                    Dont get null data for metadata bc will be treated inside fit function.
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
        self.epochs = epochs

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Loss: {loss:.3f}'
            iterator.set_description(iterator_description.format(loss=0))

        for i in iterator:
            loss_values = []
            batch = []
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
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
                'Batch': batch,
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
                "Time_of_fit": pre_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Fit_duration": str(fit_duration).split('.')[0],
                "Loss": self.loss_values
        }

        self.metadata["model"]["fit_settings"]["times_fitted"] += 1
        self.metadata["model"]["fit_settings"]["fit_history"].append(fit_dict)

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