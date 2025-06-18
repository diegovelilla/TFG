from copy import deepcopy
import torch
import numpy as np
from .utils_train import update_ema
import pandas as pd
from tqdm import tqdm

class Trainer:
    def __init__(self, diffusion, train_loader, lr, weight_decay, steps, device=torch.device('cuda:0')):
        self.diffusion = diffusion
        
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        
        self.train_loader_original = train_loader  
        self.train_loader = iter(train_loader)
        
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        # self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.loss_history = pd.DataFrame(columns=['step', 'loss'])

    def _anneal_lr(self, step):
        lr = self.init_lr * (1 - step / self.steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x):
        x = x.to(self.device)
        self.optimizer.zero_grad()
        # for unconditional training, pass an empty dictionary.
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, {})
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()
        return loss_multi, loss_gauss

    def run_loop(self, verbose):
        pbar = tqdm(range(self.steps))
        for step in pbar:

            try:
                x, _ = next(self.train_loader)
            except StopIteration:
                self.train_loader = iter(self.train_loader_original)
                x, _ = next(self.train_loader)

            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_count = 0
            
            loss_multi, loss_gauss = self._run_step(x)

            batch_size = x.size(0)
            curr_count += batch_size
            curr_loss_multi += loss_multi.item() * batch_size
            curr_loss_gauss += loss_gauss.item() * batch_size

            total_loss = loss_multi.item() + loss_gauss.item()
            pbar.set_description(f"Loss: {total_loss:.4f}")

            mloss = np.around(curr_loss_multi / curr_count, 4)
            gloss = np.around(curr_loss_gauss / curr_count, 4)
            total = mloss + gloss
            self.loss_history.loc[len(self.loss_history)] = [step, total]

            self._anneal_lr(step)
            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

        return self.loss_history
