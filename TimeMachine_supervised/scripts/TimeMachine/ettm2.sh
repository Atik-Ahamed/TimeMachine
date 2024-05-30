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

root_path_name=../data/ETT-small
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

rin=1
random_seed=2024
one=96
two=192
three=336
four=720
residual=1
fc_drop=0.6
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
                n1=256
                n2=32
            fi
            if [ $pred_len -eq $two ]
            then
                n1=128
                n2=64
            fi
            if [ $pred_len -eq $three ]
            then
                n1=256
                n2=128
            fi
            if [ $pred_len -eq $four ]
            then
                n1=256
                n2=128
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
            --enc_in 7 \
            --n1 $n1 \
            --n2 $n2 \
            --dropout $fc_drop\
            --des 'Exp' \
            --train_epochs 100\
            --itr 1 --batch_size 1024 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$n1'_'$n2'_'$fc_drop'_'$rin'_'$residual'_'$dstate'_'$dconv'_'$e_fact.log
        done      
    done
done