export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/TimeiTransformer" ]; then
    mkdir ./log/TimeiTransformer
fi

if [ ! -d "./log/TimeiTransformer/weather" ]; then
    mkdir ./log/TimeiTransformer/weather
fi

seq_len=96

model_name=TimeiTransformer


for seed in 2020 2021 2022
do
python -u run.py \
  --seed $seed \
  --task_name long_term_forecast \
  --time_feature_types HourOfDay \
  --with_curve 0 \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --rda 1 \
  --rdb 1 \
  --ksize 3 \
  --beta 0.4 \
  --itr 1 | tee -a ./log/TimeiTransformer/weather/$seq_len.txt
done

for seed in 2020 2021 2022
do
python -u run.py \
  --seed $seed \
  --task_name long_term_forecast \
  --time_feature_types MinuteOfHour HourOfDay \
  --with_curve 0 \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --rda 1 \
  --rdb 1 \
  --ksize 3 \
  --beta 0.4 \
  --itr 1 | tee -a ./log/TimeiTransformer/weather/$seq_len.txt
done

for seed in 2020 2021 2022
do
python -u run.py \
  --seed $seed \
  --task_name long_term_forecast \
  --time_feature_types MinuteOfHour HourOfDay \
  --with_curve 0 \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --rda 1 \
  --rdb 1 \
  --ksize 3 \
  --beta 0.4 \
  --itr 1 | tee -a ./log/TimeiTransformer/weather/$seq_len.txt
done

for seed in 2020 2021 2022
do
python -u run.py \
  --seed $seed \
  --task_name long_term_forecast \
  --time_feature_types MinuteOfHour HourOfDay \
  --with_curve 0 \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --rda 1 \
  --rdb 1 \
  --ksize 3 \
  --beta 0.4 \
  --itr 1 | tee -a ./log/TimeiTransformer/weather/$seq_len.txt
done
