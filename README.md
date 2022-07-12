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


## Data
- The precompressed point cloud maps can be downloaded [here](https://www.ipb.uni-bonn.de/html/projects/retriever/oxford_compressed.zip).
- For the uncompressed point clouds, I refer to the [PointNetVLAD](https://github.com/mikacuy/pointnetvlad).
