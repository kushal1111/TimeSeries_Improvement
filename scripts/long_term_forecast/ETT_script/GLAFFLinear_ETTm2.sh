export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/GLAFFLinear" ]; then
    mkdir ./log/GLAFFLinear
fi

if [ ! -d "./log/GLAFFLinear/ettm2" ]; then
    mkdir ./log/GLAFFLinear/ettm2
fi

model_name=GLAFFLinear


for seq_len in 96
do
for pred_len in 96 192 336 720
do
for seed in 2020 2021 2022
do
python -u run.py \
  --time_feature_types MonthOfYear DayOfMonth DayOfWeek HourOfDay MinuteOfHour SecondOfMinute \
  --task_name long_term_forecast \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --freq t \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --batch_size 128 \
  --train_epochs 10 \
  --num_workers 10 \
  --dropout 0.0 \
  --loss mse \
  --seed $seed \
  --itr 1 | tee -a ./log/GLAFFLinear/ettm2/$seq_len.txt
done
done
done
