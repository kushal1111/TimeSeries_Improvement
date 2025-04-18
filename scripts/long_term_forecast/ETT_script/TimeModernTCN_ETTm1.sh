export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/TimeModernTCN" ]; then
    mkdir ./log/TimeModernTCN
fi

if [ ! -d "./log/TimeModernTCN/ettm1" ]; then
    mkdir ./log/TimeModernTCN/ettm1
fi

seq_len=96

python -u run.py \
    --time_feature_types MinuteOfHour HourOfDay \
    --task_name long_term_forecast \
    --is_training 1 \
    --with_curve 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --model_id ETTm1_$seq_len'_'96 \
    --model TimeModernTCN \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --ffn_ratio 8 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 3 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.0 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --dropout 0.3 \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 512 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des Exp \
    --rda 1 \
    --rdb 1 \
    --ksize 5 \
    --beta 0.9 \
    --lradj type3 | tee -a ./log/TimeModernTCN/ettm1/$seq_len.txt


python -u run.py \
    --time_feature_types MinuteOfHour HourOfDay \
    --task_name long_term_forecast \
    --is_training 1 \
    --with_curve 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --model_id ETTm1_$seq_len'_'192 \
    --model TimeModernTCN \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --ffn_ratio 8 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 3 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --dropout 0.3 \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 512 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des Exp \
    --rda 1 \
    --rdb 1 \
    --ksize 5 \
    --beta 0.9 \
    --lradj type3 | tee -a ./log/TimeModernTCN/ettm1/$seq_len.txt

python -u run.py \
    --time_feature_types MinuteOfHour HourOfDay \
    --task_name long_term_forecast \
    --is_training 1 \
    --with_curve 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --model_id ETTm1_$seq_len'_'336 \
    --model TimeModernTCN \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --ffn_ratio 8 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 3 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --dropout 0.3 \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 512 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des Exp \
    --rda 1 \
    --rdb 1 \
    --ksize 5 \
    --beta 0.9 \
    --lradj type3 | tee -a ./log/TimeModernTCN/ettm1/$seq_len.txt

python -u run.py \
    --time_feature_types MinuteOfHour HourOfDay \
    --task_name long_term_forecast \
    --is_training 1 \
    --with_curve 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --model_id ETTm1_$seq_len'_'720 \
    --model TimeModernTCN \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --ffn_ratio 8 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 3 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --dropout 0.3 \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 512 \
    --patience 3 \
    --learning_rate 0.0001 \
    --des Exp \
    --rda 1 \
    --rdb 1 \
    --ksize 5 \
    --beta 0.9 \
    --lradj type3 | tee -a ./log/TimeModernTCN/ettm1/$seq_len.txt
