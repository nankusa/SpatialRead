config_dir=./configs/spnode_mlp/geo

for config in `ls $config_dir`; do
    config_path="$config_dir/$config"
    echo "build data with config $config_path"
    echo "RUN COMMAND: python -m spatialread.finetune train --config $config_path"
    python -m spatialread.finetune train --config $config_path
done