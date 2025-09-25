# SpatialRead

SpatialRead is a readout function to enhance modern MPNNs in spatial properties, such as gas adsorption, void fraction etc.

## Install

```sh
conda create -n spatialread python=3.10
conda activate spatialread
pip install -r ./requirements.txt

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch_geometric
```

## Data process

We provide test data in `/data/test`.

First build pyg Data

```sh
python -m spatialread.data.build_graph --config ./configs/test.yaml
```

Next, calculate `edge_index`

```sh
python -m spatialread.data.build_edge --config ./configs/test.yaml --devices cuda:0
python -m spatialread.data.build_edge --config ./configs/test.yaml --devices cuda:0 --sp
```

## Train

```sh
python -m spatialread.finetune train --config ./configs/test.yaml
```