import click
from os.path import join, dirname, abspath
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml
import retriever.datasets.datasets as datasets
import retriever.models.models as models
import retriever.models.evaluation as evaluation


@click.command()
# Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)), 'config/config.yaml'))
@click.option('--data_config',
              '-dc',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)), 'config/oxford_data.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
def main(config, data_config, weights, checkpoint):
    cfg = yaml.safe_load(open(config))
    data_cfg = yaml.safe_load(open(data_config))
    cfg['git_commit_version'] = str(subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).strip())
    cfg['data_config'] = data_cfg
    print(f"Start experiment {cfg['experiment']['id']}")

    data = datasets.getOxfordDataModule(data_cfg)

    model = models.getModel(cfg['network_architecture'], cfg, weights)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(monitor='val_top_1_acc',
                                       filename=cfg['experiment']['id'] +
                                       '_{epoch:03d}_{val_top_1_acc:.2f}',
                                       mode='max',
                                       save_last=True)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)
    validation_callback = evaluation.PreComputeLatentsCallback(
        data.val_latent_dataloader())

    print('nr gpus:', cfg['train']['n_gpus'])
    # Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      max_epochs=cfg['train']['max_epoch'],
                      callbacks=[lr_monitor, checkpoint_saver, validation_callback],)

    # Train!
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
