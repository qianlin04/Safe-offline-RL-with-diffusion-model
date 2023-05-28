cost_grad_weight=100
batch_size=64
seed=26

gpu_id=1
guide_type=1

n_test_episode=20
value_seed=5
control_interval=2

value_loadpath="f:values/defaults_seed${value_seed}_H{horizon}_T{n_diffusion_steps}_d{discount}"
cost_value_loadpath="f:vel_cost_values/defaults_seed${value_seed}_H{horizon}_T{n_diffusion_steps}_d{discount}"
for ((i=$2; i<=$3; i=i+2 ))
do 
    cw=$(echo $i | awk '{ printf "%.1f" ,$1*0.1}')
    nohup python scripts/plan_cost_guided.py --dataset $1 --n_test_episode $n_test_episode --gpu $gpu_id --guide_type $guide_type --discount 1.0 --seed $seed --ratio_of_maxthreshold $cw --cost_grad_weight $cost_grad_weight --batch_size $batch_size > ./results/$1/cost/vel_cost_discount_cw${cost_grad_weight}_seed${seed}_guide_type${guide_type}_${cw}.txt &
    #nohup python scripts/plan_cost_guided.py --dataset $1 --random_budget 1 --control_interval $control_interval --n_test_episode $n_test_episode --gpu $gpu_id --guide_type $guide_type --value_loadpath "$value_loadpath" --cost_value_loadpath "$cost_value_loadpath" --discount 0.99 --ratio_of_maxthreshold $cw --cost_grad_weight $cost_grad_weight --batch_size $batch_size > ./results/$1/cost/${control_interval}_${seed}_${cw}.txt &
done