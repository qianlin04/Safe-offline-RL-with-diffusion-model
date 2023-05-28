cost_grad_weight=100
batch_size=64
seed=19

gpu_id=2
guide_type=1

n_test_episode=100
value_seed=5
control_interval=1

value_loadpath="f:values/defaults_seed${value_seed}_H{horizon}_T{n_diffusion_steps}_d{discount}"
cost_value_loadpath="f:vel_cost_values/defaults_seed${value_seed}_H{horizon}_T{n_diffusion_steps}_d{discount}"

cw=1
# dataset="SafetyCarCircle-v0"
# python scripts/plan_cost_guided.py --dataset $dataset --random_budget 1 --control_interval $control_interval --n_test_episode $n_test_episode --gpu $gpu_id --guide_type $guide_type --discount 1.0 --ratio_of_maxthreshold $cw --cost_grad_weight $cost_grad_weight --batch_size $batch_size  > ./results/$dataset/cost/${control_interval}_${seed}_${cw}.txt 
dataset="SafetyAntRun-v0"
python scripts/plan_cost_guided.py --dataset $dataset --random_budget 1 --control_interval $control_interval --n_test_episode 82 --gpu $gpu_id --guide_type $guide_type --discount 1.0 --ratio_of_maxthreshold $cw --cost_grad_weight $cost_grad_weight --batch_size $batch_size  > ./results/$dataset/cost/${control_interval}_${seed}_${cw}.txt 
dataset="SafetyBallReach-v0"
python scripts/plan_cost_guided.py --dataset $dataset --random_budget 1 --control_interval $control_interval --n_test_episode $n_test_episode --gpu $gpu_id --guide_type $guide_type --discount 1.0 --ratio_of_maxthreshold $cw --cost_grad_weight $cost_grad_weight --batch_size $batch_size  > ./results/$dataset/cost/${control_interval}_${seed}_${cw}.txt 


