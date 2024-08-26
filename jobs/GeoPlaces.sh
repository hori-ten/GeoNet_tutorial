echo "start..."

source=$1
target=$2
data_dir= $3
queue_size=$4

nClasses=204
BatchSize=32
dataset="GeoPlaces"
out_dir=MemSAC_${source}${target}_QS_${queue_size}

python3 train.py --dataset ${dataset} --source ${source} --target ${target} --lr 0.03 --out_dir ${out_dir} --max_iteration 100001 --batch_size ${BatchSize} --data_dir $3 --total_classes ${nClasses} --multi_gpu 0 --test-iter 5000 --queue_size $4 --adv-coeff 1. --sim-coeff 0.1