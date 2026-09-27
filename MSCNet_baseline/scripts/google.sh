export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/long_term_forecast_workload" ]; then
    mkdir ./logs/long_term_forecast_workload
fi
seq_len=96
model_name=MSCNet

root_path_name=./dataset
data_path_name=google.csv
model_id_name=google
data_name=custom


for pred_len in 24 48 72 96
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --freq t \
    --seq_len $seq_len \
    --label_len $seq_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 100 \
    --dec_in 100 \
    --c_out 100 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 4 \
    --d_model 128 \
    --train_epochs 100 \
    --patience 3  > logs/long_term_forecast_workload/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

