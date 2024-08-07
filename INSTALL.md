# Installing Conda Environment from Zero to Hero

```Shell
conda create -n lfd_dp3_juno1 python=3.10 -y
conda activate lfd_dp3_juno1

pip uninstall -y torch torchvision torchaudio

# on CUDA-11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

conda install -y fvcore iopath -c conda-forge -c iopath -c fvcore -y
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html


# # on CUDA-12.1
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# conda install -y fvcore iopath -c conda-forge -c iopath -c fvcore -y
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html

cd 3D-Diffusion-Policy && pip install -e . && cd ..

pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

cd third_party
cd gym-0.21.0 && pip install -e . && cd ..
cd Metaworld && pip install -e . && cd ..

cd mujoco-py-2.1.2.14
pip install -e .
cd ../..

pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor

pip install opencv-python==4.9.0.80 opencv-python-headless==4.9.0.80
pip install -U opencv-contrib-python==4.7.0.72

pip install hydra-core wandb open3d chardet trimesh natsort ffmpeg imageio ipykernel pygame

pip install pymunk==6.6.0
pip install shapely==2.0.2

pip install scikit-learn==1.3.0 scikit-spatial==7.0.0 scikit-video==1.1.11 scikit-image==0.22.0
```
----------------------------------------------------------------------------------------------------------------------------

The following guidance works well for a machine with 3090/A40/A800/A100 GPU, cuda 11.7, driver 515.65.01.

First, git clone this repo and `cd` into it.

    git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git


**Please strictly follow the guidance to avoid any potential errors. Especially, make sure Gym version is the same.**

**Don't worry about the gym version now. Just install my version in `third_party/gym-0.21.0` and you will be fine.**

---

1.create python/pytorch env

    conda remove -n dp3 --all
    conda create -n dp3 python=3.8
    conda activate dp3


---

2.install torch

    # if using cuda>=12.1
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # else, 
    # just install the torch version that matches your cuda version

---

3.install dp3

    cd 3D-Diffusion-Policy && pip install -e . && cd ..


---

4.install mujoco in `~/.mujoco`

    cd ~/.mujoco
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

    tar -xvzf mujoco210.tar.gz

and put the following into your bash script (usually in `YOUR_HOME_PATH/.bashrc`). Remember to `source ~/.bashrc` to make it work and then open a new terminal.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    export MUJOCO_GL=egl


and then install mujoco-py (in the folder of `third_party`):

    cd YOUR_PATH_TO_THIRD_PARTY
    cd mujoco-py-2.1.2.14
    pip install -e .
    cd ../..


----

5.install sim env

    pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

    cd third_party
    cd dexart-release && pip install -e . && cd ..
    cd gym-0.21.0 && pip install -e . && cd ..
    cd Metaworld && pip install -e . && cd ..
    cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ..

download assets from [Google Drive](https://drive.google.com/file/d/1DxRfB4087PeM3Aejd6cR-RQVgOKdNrL4/view?usp=sharing), unzip it, and put it in `third_party/dexart-release/assets`. 

download Adroit RL experts from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), unzip it, and put the `ckpts` folder under `$YOUR_REPO_PATH/third_party/VRL3/`.

---

6.install pytorch3d (a simplified version)

    cd third_party/pytorch3d_simplified && pip install -e . && cd ..


---

7.install some necessary packages

    pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor


---

8.install our visualizer for pointclouds (optional)

    pip install kaleido plotly
    cd visualizer && pip install -e . && cd ..

