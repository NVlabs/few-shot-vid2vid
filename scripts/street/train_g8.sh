python train.py --name street --dataset_mode fewshot_street \
--adaptive_spade --loadSize 512 --fineSize 512 \
--gpu_ids 0,1,2,3,4,5,6,7 --batchSize 46 --nThreads 16 --continue_train
