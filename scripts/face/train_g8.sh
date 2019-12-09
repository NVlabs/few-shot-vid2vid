python train.py --name face --dataset_mode fewshot_face \
--adaptive_spade --warp_ref --spade_combine \
--gpu_ids 0,1,2,3,4,5,6,7 --batchSize 60 --nThreads 16 --continue_train