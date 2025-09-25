config_dir=./configs/base

for config in `ls $config_dir`; do
    config_path="$config_dir/$config"
    echo "build data with config $config_path"
    echo "RUN COMMAND: python -m spatialread.data.build_graph --config $config_path"
    python -m spatialread.data.build_graph --config $config_path
    echo "RUN COMMAND: python -m spatialread.data.build_edge --config $config_path"
    python -m spatialread.data.build_edge --config $config_path --devices cuda:1,cuda:2 --sp
done