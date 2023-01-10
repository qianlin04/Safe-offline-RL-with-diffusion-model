cost_grad_weight=1000
gpu_id=1
guide_type=1
seed=5
horizon=32

batch_size=64
n_test_episode=20
#value_loadpath="f:values/defaults_seed${seed}_H{horizon}_T{n_diffusion_steps}_d{discount}"
#cost_value_loadpath="f:vel_cost_values/defaults_seed${seed}_H{horizon}_T{n_diffusion_steps}_d{discount}"
for ((i=$2; i<=$3; i=i+1 ))
do 
    cw=$(echo $i | awk '{ printf "%.1f" ,$1*0.1}')
    nohup python scripts/plan_cost_guided.py --config config.toy_safe_env --n_test_episode $n_test_episode --dataset $1 --guide_type $guide_type --gpu $gpu_id --horizon $horizon --discount 0.99 --seed $seed --ratio_of_maxthreshold $cw --cost_grad_weight $cost_grad_weight --batch_size $batch_size > ./results/$1/cost/vel_cost_discount_cw${cost_grad_weight}_seed${seed}_horizon${horizon}_${cw}.txt &
done