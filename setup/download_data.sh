#!/bin/bash

# Navigate to the habitat-lab_soyeonm directory
cd habitat-lab_soyeonm

# Download specific datasets using habitat_sim utilities
#python -m habitat_sim.utils.datasets_download --uids humanoid_data --data-path data/
python -m habitat_sim.utils.datasets_download --uids ycb hab_fetch hab_spot_arm replica_cad_dataset rearrange_pick_dataset_v0 rearrange_dataset_v1 --data-path data/

# Navigate to the data directory
cd data

# Install Git Large File Storage (LFS) - check for installation and install if not present
if ! command -v git-lfs &> /dev/null
then
    echo "Git LFS could not be found, attempting to install..."
    # This installation line might need to be adapted depending on the OS and package manager
    sudo apt-get install git-lfs
fi

# Set up Git LFS for your user account
git lfs install

# Clone the dataset repository from Hugging Face
git clone https://huggingface.co/datasets/fpss/fphab
git checkout eb39f81cf2fd041a7035ccfef6885bcd51b6b0cd 

# Rename the cloned directory
mv fphab fpss

wget --no-check-certificate  'https://drive.google.com/uc?export=download&id=1h4WpS0WhE6ytEuP3imH4tZMz1A-J-ch6' -O humanoids.zip 
unzip humanoids.zip 
rm -r humanoids.zip
cd humanoids/humanoid_data
cp -r * ../../
cd ../../


cd datasets

wget --no-check-certificate  'https://drive.google.com/uc?export=download&id=1JHIUu3Lzf59SoGOHGmHCqNzZZ0hJLZlL' -O sif_release.zip 
unzip sif_release.zip 
rm -r sif_release.zip

cd $SIF_DIR
wget --no-check-certificate  'https://drive.google.com/uc?export=download&id=1SB8zuSRC4NCFRyXv9jeMj1W95Lne3Ja1' -O fbe_mapping_actions.zip 
unzip fbe_mapping_actions.zip 
rm -r fbe_mapping_actions.zip

wget --no-check-certificate  'https://drive.google.com/uc?export=download&id=182S3PZORoCMPB5VvU5ACiVeHQyxZttFb' -O task_load_folder.zip 
unzip task_load_folder.zip 
rm -r task_load_folder.zip

ln -s $HOME_DIR/habitat-lab_soyeonm/data $SIF_DIR/data 

echo "Setup completed."