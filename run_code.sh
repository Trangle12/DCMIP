CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py \
-d msmt17 --eps 0.7 --epochs_stage1 40 --epochs_stage2 50 --iters_stage1 100 --iters_stage2 100 \
--logs-dir ../logs/msmt17


CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py \
-d market1501 --eps 0.45 --epochs_stage1 40 --epochs_stage2 50 --iters_stage1 100 --iters_stage2 100 \
--logs-dir ../logs/market1501


CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py \
-d dukemtmcreid --eps 0.7 --epochs_stage1 40 --epochs_stage2 50 --iters_stage1 100 --iters_stage2 100 \
--logs-dir ../logs/dukemtmcreid