from abc import abstractmethod
import torch
import torch.optim.lr_scheduler
import torch.nn as nn
import retriever.models.blocks as blocks
import retriever.models.loss as pnloss
from math import pi
from pytorch_lightning.core.lightning import LightningModule
import numpy as np


def getModel(model_name: str, config: dict, weights: str = None):
    """Returns the model with the specific model_name. 

    Args:
        model_name ([str]): Name of the architecture (should be a LightningModule)
        config ([dict]): Parameters of the model
        weights ([str], optional): [description]. if specified: loads the weights

    Returns:
        [type]: [description]
    """
    if weights is None:
        return eval(model_name)(config)
    else:
        print(weights)
        return eval(model_name).load_from_checkpoint(weights, hparams=config)

##################################
# Base Class
##################################


class PNPerceiverModule(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
       # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)

        self.pnloss = pnloss.LazyQuadrupletLoss(
            margin_1=hparams['loss']['margin_1'],
            margin_2=hparams['loss']['margin_2'],
            lazy=hparams['loss']['lazy'])

        # Networks
        self.model = nn.Sequential(
            blocks.PointNetFeat(
                norm=hparams['model']['norm'], in_dim=hparams['point_net']['input_dim'], out_dim=hparams['model']['input_dim']),
            PerceiverEncoder(**hparams['model']),
            nn.Linear(hparams['model']['latent_dim'], hparams['net_vlad']['in_dim']))

        self.vladnet = blocks.PerceiverVLAD(
            num_latents=hparams['model']['num_latents'], in_dim=hparams['net_vlad']['in_dim'], out_dim=hparams['net_vlad']['out_dim'])

        self.top_k = [1, 5, 10]
        self.latents = None
        self.latents_idx = None

    def forward(self, x):
        x = self.model(x)
        x = self.vladnet(x)
        if isinstance(x, list):
            x = torch.stack(x, dim=-2)
        return x

    def getLoss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, second_negative: torch.Tensor = None):
        return self.pnloss(anchor, positive, negative, second_negative)

    def on_validation_end(self):
        self.latents = None  # Reset latents
        self.latents_idx = None

    def training_step(self, batch: dict, batch_idx):
        query = self.forward(batch['query'])
        positives = self.forward(batch['positives'])
        negatives = self.forward(batch['negatives'])
        neg2 = self.forward(batch['neg2'])

        loss = self.getLoss(query, positives, negatives, neg2)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: dict, batch_idx):
        query = self.forward(batch['query'])
        positives = self.forward(batch['positives'])
        negatives = self.forward(batch['negatives'])
        neg2 = self.forward(batch['neg2'])
        loss = self.getLoss(query, positives, negatives, neg2)
        self.log('val_loss', loss)

        dist = (query - self.latents.unsqueeze(0)).pow(2).sum(dim=-1)
        # k+1, to not count itself
        best, indices = dist.topk(np.max(self.top_k)+1, 1, largest=False)
        should = batch['is_pos']

        for k in self.top_k:
            best_idx = self.latents_idx[indices[:, :k+1]]
            correct, _ = torch.gather(should, 1, best_idx).max(dim=1)
            accuracy = correct.sum()/correct.shape[0]
            self.log(f'val_top_{k}_acc', accuracy)
        return loss

    def save_query_step(self, batch: dict):
        latents = self.forward(batch['query'].to(self.device)).squeeze()
        latents_idx = batch['query_idx'].to(self.device)
        if self.latents is None:
            self.latents = latents
            self.latents_idx = (latents_idx)
        else:
            self.latents = torch.vstack([self.latents, latents])
            self.latents_idx = torch.hstack([self.latents_idx, latents_idx])

    def test_step(self, batch: dict, batch_idx):
        query = self.forward(batch['query'])
        positives = self.forward(batch['positives'])
        negatives = self.forward(batch['negatives'])
        neg2 = self.forward(batch['neg2'])

        loss = self.getLoss(query, positives, negatives, neg2)
        self.log('test:loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams['train']['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
        return [optimizer], [scheduler]


#######################################################################################
######################### Perceiver ###################################################
#######################################################################################

class PerceiverEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 num_latents=0,
                 num_attents=1,
                 num_transf_per_attent=4,
                 dropout=0.5,
                 num_heads=1,
                 norm=True):
        """A Perceiver encoder composed of a cross attention and several self attention blocks.

        Args:
            input_dim (int): Dimension of input
            latent_dim (int): Dimension of latent vectors
            num_latents (int, optional): number of latents. Defaults to 0.
            num_attents (int, optional): num of recurrency. Defaults to 1.
            num_transf_per_attent (int, optional): num of self attention layer after the cross attention layer. Defaults to 4.
            dropout (float, optional): dropout rate. Defaults to 0.0.
            num_heads (int, optional): num heads for the multihead attention. Defaults to 1.
        """
        super().__init__()
        self.latents = nn.Parameter(torch.randn(
            1, num_latents, latent_dim), requires_grad=True)
        self.cross_attn = blocks.PerceiverBlock(latent_dim=latent_dim,
                                                inp_dim=input_dim,
                                                dropout=dropout,
                                                num_heads=num_heads)
        self.latent_attn = [blocks.PerceiverBlock(latent_dim=latent_dim,
                                                  dropout=dropout,
                                                  num_heads=num_heads) for _ in range(num_transf_per_attent)]
        self.latent_attn = nn.Sequential(*self.latent_attn)
        self.num_attents = num_attents
        if self.num_attents > 0:  # Weight sharing for all the blocks after the first one
            self.cross_attn_shared = blocks.PerceiverBlock(latent_dim=latent_dim,
                                                           inp_dim=input_dim,
                                                           dropout=dropout,
                                                           num_heads=num_heads)
            self.latent_attn_shared = nn.Sequential(*[blocks.PerceiverBlock(latent_dim=latent_dim,
                                                                            dropout=dropout,
                                                                            num_heads=num_heads) for _ in range(num_transf_per_attent)])

    def forward(self, input: torch.Tensor, latents: torch.Tensor = None):
        """Recurrently transforms the latent features based on the input data.

        Args:
            input (torch.Tensor): Input features
            latents (torch.Tensor, optional): The latent variables to update. Takes the own parameters if None. Defaults to None.

        Returns:
            [type]: [description]
        """
        latents = self.latents if latents is None else latents
        latents = self.cross_attn(latents, input)
        latents = self.latent_attn(latents)
        for _ in range(self.num_attents):
            latents = self.cross_attn_shared(latents, input)
            latents = self.latent_attn_shared(latents)
        return latents

