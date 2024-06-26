# Clone habitat-sim and install
cd ..
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout 366294cadd914791e57d7d70e61ae67026386f0b
pip install -r requirements.txt
python setup.py install --headless --with-cuda --bullet
cd ..

# Clone habitat-lab_soyeonm and install
git clone https://github.com/soyeonm/habitat-lab_soyeonm.git
cd habitat-lab_soyeonm
git checkout SIRO_pointgoal
pip install -e habitat-lab 
pip install -e habitat-baselines 
cd ..

# Install PyTorch, torchvision, torchaudio, and detectron2
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# Clone and install Detectron2 from its repository
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..

# Clone and setup Detic
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
export  Detic_directory=$(pwd)
pip install -r requirements.txt
cd ..

pip install trimesh
