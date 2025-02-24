export CUDA_VISIBLE_DEVICES=0

model_name=TimeiTransformer

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/TimeiTransformer" ]; then
    mkdir ./log/TimeiTransformer
fi

if [ ! -d "./log/TimeiTransformer/ecl" ]; then
    mkdir ./log/TimeiTransformer/ecl
fi

seq_len=96

for seed in 2020 2021 2022
do
python -u run.py \
  --with_curve 0 \
  --seed $seed \
  --task_name long_term_forecast \
  --time_feature_types HourOfDay DayOfWeek  \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --num_workers 1 \
  --learning_rate 0.001 \
  --rda 1 \
  --rdb 1 \
  --ksize 5 \
  --beta 0.1 \
  --itr 1 | tee -a ./log/TimeiTransformer/ecl/$seq_len.txt
done

for seed in 2020 2021 2022
do
python -u run.py \
  --with_curve 0 \
  --seed $seed \
  --task_name long_term_forecast \
  --time_feature_types HourOfDay DayOfWeek \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --num_workers 1 \
  --learning_rate 0.001 \
  --rda 1 \
  --rdb 1 \
  --ksize 5 \
  --beta 0.1 \
  --itr 1 | tee -a ./log/TimeiTransformer/ecl/$seq_len.txt
done

for seed in 2020 2021 2022
do
python -u run.py \
  --with_curve 0 \
  --seed $seed \
  --task_name long_term_forecast \
  --time_feature_types HourOfDay DayOfWeek SeasonOfYear \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --num_workers 1 \
  --learning_rate 0.0005 \
  --rda 1 \
  --rdb 1 \
  --ksize 5 \
  --beta 0.1 \
  --itr 1 | tee -a ./log/TimeiTransformer/ecl/$seq_len.txt
done

for seed in 2020 2021 2022
do
python -u run.py \
  --with_curve 0 \
  --seed $seed \
  --task_name long_term_forecast \
  --time_feature_types HourOfDay DayOfWeek SeasonOfYear \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --num_workers 1 \
  --learning_rate 0.0005 \
  --rda 1 \
  --rdb 1 \
  --ksize 5 \
  --beta 0.1 \
  --itr 1 | tee -a ./log/TimeiTransformer/ecl/$seq_len.txt
done
