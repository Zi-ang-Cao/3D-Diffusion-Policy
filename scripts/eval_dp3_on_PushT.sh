# use the same command as training except the script
# for example:
# bash scripts/eval_dp3_on_PushT.sh dp3 pusht_cornerpc 0806_3am 1 0
# bash scripts/eval_dp3_on_PushT.sh simple_dp3 pusht_cornerpc 0806_5pm 1 0



DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="3D-Diffusion-Policy/logs_ac_dim3/train_${alg_name}/${exp_name}_seed${seed}"


gpu_id=${5}


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



                                