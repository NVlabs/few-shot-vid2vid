python train.py --name pose --dataset_mode fewshot_pose \
--adaptive_spade --warp_ref --spade_combine --remove_face_labels --add_face_D \
--niter_single 100 --niter 200 \
--gpu_ids 0,1,2,3,4,5,6,7 --batchSize 30 --nThreads 16 --continue_train 