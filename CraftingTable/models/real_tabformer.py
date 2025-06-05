from realtabformer import REaLTabFormer
from datetime import datetime
from .base import BaseModel
from typing import Any, Dict
import pandas as pd
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset
from torch import nn
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.optimization import get_scheduler


class SaveEpochEndCallback(TrainerCallback):
    """This callback forces a checkpoint save at each epoch end."""

    def __init__(self, save_epochs: int = None) -> None:
        super().__init__()

        self.save_epochs = save_epochs

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.save_epochs is not None:
            control.should_save = math.ceil(state.epoch) % self.save_epochs == 0
        else:
            control.should_save = True

        return control
    
class ResumableTrainer(Trainer):
    """This trainer makes the scheduler consistent over pauses
    in the training. The scheduler should return values similar
    to when a training is done either intermittently or continuously
    over the `target_epochs`.
    """

    def __init__(
        self,
        target_epochs: int = None,
        save_epochs: int = None,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = None,
    ):
        # Declare here for typing
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None

        if callbacks is None:
            callbacks = []

        callbacks.append(SaveEpochEndCallback(save_epochs=save_epochs))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.target_epochs = target_epochs

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """

        if self.lr_scheduler is None:
            if self.target_epochs is not None:
                # Compute the max_steps based from the
                # `target_epochs`.
                train_dataloader = self.get_train_dataloader()
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = (
                    len_dataloader // self.args.gradient_accumulation_steps
                )
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

                max_steps = math.ceil(self.target_epochs * num_update_steps_per_epoch)
                num_training_steps = max_steps

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

        return self.lr_scheduler

class RealTabformer(REaLTabFormer, BaseModel):

    def __init__(self):
        BaseModel.__init__(self)

    def __repr__(self):
        return f"RealTabformer(data_type='tabular', model='GPT-2')"
    
    def fit(self, 
        train_data: pd.DataFrame,
        discrete_columns: list[str],
        device: str = "cuda",
        epochs: int = 1000, 
        batch_size: int = 64,
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = 0,
        mask_rate: float = 0):
        REaLTabFormer.__init__(self, model_type="tabular", epochs=epochs, batch_size=batch_size,
                               early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold,
                               mask_rate=mask_rate)
        
        self.metadata['model']['hyperparmeters'] = "Uses GPT-2 model hyperparameters."

        self.data = pd.DataFrame(self._transform_data(data=train_data, discrete_columns=discrete_columns))
        pre_time = datetime.now()
        REaLTabFormer.fit(self, self.data)
        post_time = datetime.now()
        fit_duration = post_time - pre_time
        fit_dict = {
                    "time_of_fit": pre_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": str(fit_duration).split('.')[0],
                    "hyperparameters": {"device": device,
                                        "epochs": epochs,
                                        "batch_size": batch_size,
                                        "early_stopping_patience": early_stopping_patience,
                                        "early_stopping_threshold": early_stopping_threshold,
                                        "mask_rate": mask_rate},
                    "loss": ""
        }
        self.metadata["model"]["fit_settings"]["times_fitted"] += 1
        self.metadata["model"]["fit_settings"]["fit_history"].append(fit_dict)

    def _build_tabular_trainer(
        self,
        device="cuda",
        num_train_epochs: int = None,
        target_epochs: int = None,
    ) -> Trainer:
        device = torch.device(device)

        # Set TrainingArguments and the Trainer
        training_args_kwargs: Dict[str, Any] = dict(self.training_args_kwargs)

        default_args_kwargs = dict(
            fp16=(
                device == torch.device("cuda")
            ),  # Use fp16 by default if using cuda device
        )

        for k, v in default_args_kwargs.items():
            if k not in training_args_kwargs:
                training_args_kwargs[k] = v

        if num_train_epochs is not None:
            training_args_kwargs["num_train_epochs"] = num_train_epochs

        # # NOTE: The `ResumableTrainer` will default to its original
        # # behavior (Trainer) if `target_epochs`` is None.
        # # Set the `target_epochs` to `num_train_epochs` if not specified.
        # if target_epochs is None:
        #     target_epochs = training_args_kwargs.get("num_train_epochs")

        callbacks = None
        if training_args_kwargs["load_best_model_at_end"]:
            callbacks = [
                EarlyStoppingCallback(
                    self.early_stopping_patience, self.early_stopping_threshold
                )
            ]

        assert self.dataset
        training_args_kwargs.pop("evaluation_strategy", None)
        trainer = ResumableTrainer(
            target_epochs=target_epochs,
            save_epochs=None,
            model=self.model,
            args=TrainingArguments(**training_args_kwargs),
            data_collator=None,
            callbacks=callbacks,
            **self.dataset,
        )

        return trainer