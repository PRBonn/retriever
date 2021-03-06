# Retriever
Point Cloud-based Place Recognition in Compressed Map

## Installation
1. Install all requirements: `pip install -r requirements.txt`
2. Install this repository: `pip install -e .`

## Usage

### Training

All the following commands should be run in `retriever/`

- Please update the config files (especially the `oxford_data.yaml` to match your data_dir)
- Run the training: `python train.py`
- The output will be saved in `retriever/experiments/{EXPERIMENT_ID}`

### Testing

- Test the model by running: `python test.py --checkpoint {PATH/TO/CHECKPOINT.ckpt} --dataset {DATASET} --base_dir {PATH/TO/DATA}`, where `{DATASET}` is e.g. `oxford`
- The output will be saved in the same folder of the checkpoint
- All the results can be visualized with: `python scripts/vis_results.py`
- The numbers of the paper are in `experiments/perceiver_pn/default/version_15/checkpoints/oxford_evaluation_query.txt`
- The pretrained model can be downloaded here [here](https://www.ipb.uni-bonn.de/html/projects/retriever/perceiver_pn_epoch=119_val_top_1_acc=0.97.ckpt) and should be placed into `experiments/perceiver_pn/default/version_15/checkpoints/`.


## Data
- The precompressed point cloud maps can be downloaded [here](https://www.ipb.uni-bonn.de/html/projects/retriever/oxford_compressed.zip).
- For the uncompressed point clouds, I refer to the [PointNetVLAD](https://github.com/mikacuy/pointnetvlad).

## Citation

If you use this library for any academic work, please cite the original paper.

```bibtex
@inproceedings{wiesmann2022icra,
author = {L. Wiesmann and R. Marcuzzi and C. Stachniss and J. Behley},
title = {{Retriever: Point Cloud Retrieval in Compressed 3D Maps}},
booktitle = {Proc.~of the IEEE Intl.~Conf.~on Robotics \& Automation (ICRA)},
year = 2022,
}
```
