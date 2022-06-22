id=$1
limit=$2
script=${@: 3}
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)

while [ $free_mem -lt $limit ]; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
    sleep 5
done

CUDA_VISIBLE_DEVICES=$id $script