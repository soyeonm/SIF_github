# Situated Instruction Following

**Situated Instruction Following** <br />
So Yeon Min, Xavi Puig, Devendra Singh Chaplot, Tsung-Yen Yang, Akshara Rai, Priyam Parashar, Ruslan Salakhutdinov, Yonatan Bisk, and Roozbeh Mottaghi<br />
Carnegie Mellon University, FAIR (Meta)

<img src="https://github.com/soyeonm/SIF/assets/77866067/75ece4c7-5996-4949-b01b-ceb83fa11dec" alt="alt text" width="600" >

[Project Webpage](https://soyeonm.github.io/SIF_webpage/)

### This repository contains:
- SIF tasks
- The code to evaluate Reasoner (LLM + mapping baseline) on the SIF task.
- We provide code of the model, and users can use their own API keys.

We **DO NOT** have any model parameters.<br />
We **ONLY** provide code and data to run upon the user providing their own OpenAI API KEY. Please use your own OpenAI API key (instructions below).<br />

## Installation
Please run
```
conda create -n sif_env python=3.9 cmake=3.14.0 -y
conda activate sif_env

git clone https://github.com/soyeonm/SIF_github.git
export HOME_DIR=$(pwd)
cd SIF_github
export SIF_DIR=$(pwd)
sh setup/install.sh
```

## Data Setup
Please have your [huggingface datasets](https://huggingface.co/datasets) username and read token ready. Now, run
```
sh setup/download_data.sh
```
to download SIF data. You will be prompted to input your hugging face username and read token (as password) when 
```
git clone https://huggingface.co/datasets/fpss/fphab
```
this line runs in the above file.

This will make your data directory
```
SIF/
  data/
    datasets/
      sif_release/
        jsons/
          pnp_val_seen.json.gz
          ...
        room_annotations/
          ...
```

Now, to download object assets, do the following (taken from habitat repo).

1. Download [Amazon and Google object archives](https://drive.google.com/drive/folders/1x6i3sDYheCWoi59lv27ZyPG4Ii2GhEZB)
2. Extract these two object datasets into `habitat-lab/data` as follows:
```
cd objects
tar -xvf ~/Downloads/google_object_dataset.tar.gz
tar -xvf ~/Downloads/amazon_berkeley.tar.gz
```

You will see your data directory like this:
```
SIF/
  data/
    armatures/
    fpss/
       hssd-hab.scene_dataset_config.json
       ...
    humanoids/
    objects/
       google_object_dataset/
       amazon_berkeley/
       ycb/
    robots/
    versioned_data/
    default.physics_config.json 
    ...
```

## Test your setup
To test that your setup is correct, run 
```
export MAGNUM_LOG=quiet   
export HABITAT_SIM_LOG=quiet 
python main.py --task_config config/s_obj/val_seen_50.yaml  --print_images 1 --exp_name  test --gt_sem_seg --magic_man_if_exists  --eps_to_run ''  --run_full_task --oracle_baseline
```
and see that it runs. 

## Add your LLM API key
We DO NOT provide the model parameters itself. To use Reasoner with Open AI's API, please add
```
export OPENAI_API_KEY= #your api key
```
to ~/.bashrc, and run 
```
source  ~/.bashrc
```

## Usage

### For evaluation: 
For running inference with Reasoner, with OpenAI API:
```
export MAGNUM_LOG=quiet   
export HABITAT_SIM_LOG=quiet 
HABITAT_SIM_LOG=quiet python main.py --task_config config/june_3/s_obj/val_seen_50.yaml  --print_images 1 --exp_name reasoner_vs_s_obj --llm_type openai --gt_sem_seg --magic_man_if_exists    --eps_to_run ''  --run_full_task 
```

Argument explanation: 

**Arguments**

--task_config: The config to use. 

--exp_name: The saving directory of model output ("tmp/dump/exp_name").

--llm_type: LLM used for inference (openai/ openai_chat). Use openai for Reasoner and openai_chat for prompter.

--print_images: Show image visualization. 

To run with GT semantic segmenation, use " --gt_sem_seg". To run with GT manipulation, use "--magic_man_if_exists". Use both for "Oracle Perception" in Table 6 of the paper, and neither for "Learned Perception".

To run with Prompter (instead of Reasoner), add "--prompter_baseline".

e.g.
```
export MAGNUM_LOG=quiet   
export HABITAT_SIM_LOG=quiet 
HABITAT_SIM_LOG=quiet python main.py --task_config config/june_3/s_obj/val_seen_50.yaml  --print_images 1 --exp_name reasoner_vs_s_obj --llm_type openai --gt_sem_seg --magic_man_if_exists    --eps_to_run ''  --run_full_task --prompter_baseline
```

To run with oracle policy, add "--oracle_baseline".

Logs and visualiations are saved to "tmp/dump/exp_name":
![Vis-0208](https://github.com/soyeonm/SIF_github/assets/77866067/e3a5a703-0fe1-4ce7-9d75-b4e2a099be81)




## Acknowledgements
This repository uses [Habitat 3.0](https://github.com/facebookresearch/habitat-lab) implementation for simulating human and spot robot trajectory.
