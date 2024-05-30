if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./csv_results" ]; then
    mkdir ./csv_results
fi
if [ ! -d "./results" ]; then
    mkdir ./results
fi
if [ ! -d "./test_results" ]; then
    mkdir ./test_results
fi
model_name=TimeMachine

root_path_name=../data/weather
data_path_name=weather.csv
model_id_name=weather
data_name=custom

rin=1
random_seed=2024
one=96
two=192
three=336
four=720
residual=1
fc_drop=0.1
dstate=256
dconv=2
for seq_len in 96
do
    for pred_len in 96 192 336 720
    do  
        for e_fact in 1
        do

            if [ $pred_len -eq $one ]
            then
                n1=128
                n2=16
                fc_drop=0.1
            fi
            if [ $pred_len -eq $two ]
            then
                n1=512
                n2=128
                fc_drop=0.5
            fi
            if [ $pred_len -eq $three ]
            then
                n1=128
                n2=64
                fc_drop=0.0
            fi
            if [ $pred_len -eq $four ]
            then
                n1=512
                n2=256
                fc_drop=0.0
            fi
            python -u run_longExp.py \
            --random_seed $random_seed \
            --is_training 1 \
            --root_path $root_path_name \
            --data_path $data_path_name \
            --model_id $model_id_name_$seq_len'_'$pred_len \
            --model $model_name \
            --data $data_name \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 21 \
            --n1 $n1 \
            --n2 $n2 \
            --dropout $fc_drop\
            --revin 1\
            --ch_ind 1\
            --residual $residual\
            --dconv $dconv \
            --d_state $dstate\
            --e_fact $e_fact\
            --des 'Exp' \
            --lradj 'constant'\
            --pct_start 0.2\
            --itr 1 --batch_size 512 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$n1'_'$n2'_'$fc_drop'_'$rin'_'$residual'_'$dstate'_'$dconv'_'$e_fact.log 
        
        done        
    done
done
