from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
import torch
# import pytimer

class PreComputeLatentsCallback(Callback):
    def __init__(self, validation_set):
        self.validation_set = validation_set

    def on_validation_start(self, trainer, pl_module: LightningModule):
        with torch.no_grad():
            for batch in self.validation_set:
                pl_module.save_query_step(batch)
