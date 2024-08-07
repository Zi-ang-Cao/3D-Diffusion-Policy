# Examples:
################# ac_dim2 #################
# In the tmux `dp3` @ bohg-ws-17
# bash scripts/train_dp3_on_PushT.sh dp3 pusht_cornerpc 0806_3am 1 0    # juno2
# bash scripts/train_dp3_on_PushT.sh dp3 pusht_cornerpc 0806_3am 2 0    # ws-17
# bash scripts/train_dp3_on_PushT.sh dp3 pusht_cornerpc 0806_3am 3 0    # ws-20

# In the tmux `dp3` @ bohg-ws-17
# bash scripts/train_dp3_on_PushT.sh simple_dp3 pusht_cornerpc 0806_5pm 1 0    # ws-20
# bash scripts/train_dp3_on_PushT.sh simple_dp3 pusht_cornerpc 0806_5pm 2 0    # Juno2
# bash scripts/train_dp3_on_PushT.sh simple_dp3 pusht_cornerpc 0806_5pm 3 0    # ws-20  [DONE]


################# ac_dim3 #################
# bash scripts/train_dp3_on_PushT.sh dp3 pusht_cornerpc 0807_2am 1 0    # juno2
# bash scripts/train_dp3_on_PushT.sh dp3 pusht_cornerpc 0807_2am 2 0    # juno2
# bash scripts/train_dp3_on_PushT.sh dp3 pusht_cornerpc 0807_2am 3 0    # gcp

# In the tmux `dp3` @ bohg-ws-17
# bash scripts/train_dp3_on_PushT.sh simple_dp3 pusht_cornerpc 0806_11pm 1 0; bash scripts/train_dp3_on_PushT.sh simple_dp3 pusht_cornerpc 0806_11pm 2 0; bash scripts/train_dp3_on_PushT.sh simple_dp3 pusht_cornerpc 0806_11pm 3 0;    # Juno2

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="3D-Diffusion-Policy/logs_ac_dim3/train_${alg_name}/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                