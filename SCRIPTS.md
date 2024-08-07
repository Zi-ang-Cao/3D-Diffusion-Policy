

```Shell
python -m mm_lfd.envs.sim_pusht.utils.process_data --in_dir /juno/u/jingyuny/projects/p_chain/data/pusht/pusht_cchi_v7_replay.zarr --out_dir /juno/u/ziangcao/Juno_CodeBase/IPRL_codeBase/Vision_Pipeline/data/pusht/processed_1211_pc2_absac_d200 --num_demos 200 --obs_mode pc2 --ac_mode abs


python -m diffusion_policy_3d.env.pushT.sim_pusht.utils.process_data_for_dp3 --in_dir /juno/u/jingyuny/projects/p_chain/data/pusht/pusht_cchi_v7_replay.zarr --num_demos 200 --obs_mode pc2 --ac_mode abs
```