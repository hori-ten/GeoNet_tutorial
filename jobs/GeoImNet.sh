echo "start..."

source=$1
target=$2
data_dir= $3

nClasses=600
BatchSize=32
dataset="GeoImNet"
out_dir=${source}${target}

python3 train.py --dataset ${dataset} --source ${source} --target ${target} --lr 0.03 --out_dir ${out_dir} --max_iteration 100000 --batch_size ${BatchSize} --data_dir $3 --total_classes ${nClasses} --multi_gpu 0 --test-iter 5000