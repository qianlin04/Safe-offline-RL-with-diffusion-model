#nohup python scripts/train.py --dataset $1 > results/$1/train.txt &
#nohup python scripts/train_cost_values.py --dataset $1 --discount 1.0 --seed 4 > results/$1/train_cost_value.txt &
nohup python scripts/train_values.py --dataset $1 --seed 4 > results/$1/train_value.txt &