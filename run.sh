CUDA_VISIBLE_DEVICES=0 python -u pretrain.py \
--data_dir ./data/baby/ \
--seg_templates_len 5 \
--cuda \
--batch_size 64 \
--lambda_scale 0.05 \
--checkpoint ./checkpoint/baby/ \
--lr 0.0005 \

CUDA_VISIBLE_DEVICES=0 python -u seq.py \
--data_dir ./data/baby/ \
--cuda \
--batch_size 128 \
--checkpoint ./checkpoint/baby/ \

CUDA_VISIBLE_DEVICES=0 python -u exp.py \
--data_dir ./data/baby/ \
--cuda \
--batch_size 128 \
--checkpoint ./checkpoint/baby/ \

CUDA_VISIBLE_DEVICES=0 python -u topn.py \
--data_dir ./data/baby/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/baby/ \
