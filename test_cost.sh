cost_grad_weight=30
batch_size=128
seed=6
gpu_id=1
value_loadpath="f:values/defaults_seed${seed}_H{horizon}_T{n_diffusion_steps}_d{discount}"
cost_value_loadpath="f:vel_cost_values/defaults_seed${seed}_H{horizon}_T{n_diffusion_steps}_d{discount}"
for ((i=$2; i<=$3; i=i+1 ))
do 
    cw=$(echo $i | awk '{ printf "%.1f" ,$1*0.1}')
    nohup python scripts/plan_cost_guided.py --dataset $1 --gpu $gpu_id --value_loadpath "$value_loadpath" --cost_value_loadpath "$cost_value_loadpath" --discount 0.99 --seed $seed --ratio_of_maxthreshold $cw --cost_grad_weight $cost_grad_weight --batch_size $batch_size > ./results/$1/cost/vel_cost_discount_cw${cost_grad_weight}_seed${seed}_${cw}.txt &
done