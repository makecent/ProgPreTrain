id=$1
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)

while [ $free_mem -lt 30000 ]; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
    sleep 5
done

CUDA_VISIBLE_DEVICES=$id bash train.sh configs/prog/i3d_prog_16x4_k400.py 1 --cfg-options data.videos_per_gpu=32