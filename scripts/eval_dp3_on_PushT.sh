# use the same command as training except the script
# for example:
# AC_DIM=2
# bash scripts/eval_dp3_on_PushT.sh dp3 pusht_cornerpc 2 0806_3am 1 0
# bash scripts/eval_dp3_on_PushT.sh simple_dp3 pusht_cornerpc 2 0806_5pm 1 0

# AC_DIM=3
# bash scripts/eval_dp3_on_PushT.sh dp3 pusht_cornerpc 3 0807_2am 1 0
# bash scripts/eval_dp3_on_PushT.sh simple_dp3 pusht_cornerpc 3 0806_11pm 1 0



DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
ac_dim=${3}
addition_info=${4}
seed=${5}
gpu_id=${6}

exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="3D-Diffusion-Policy/logs_ac_dim${ac_dim}/train_${alg_name}/${exp_name}_seed${seed}"


cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python eval_pushT.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                