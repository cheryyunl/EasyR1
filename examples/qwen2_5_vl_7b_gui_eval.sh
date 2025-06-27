#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/code/EasyR1/checkpoints/easy_r1/global_step_140/actor/huggingface # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config_eval.yaml \
    data.train_files=/code/android_control/data@validation \
    data.val_files=/code/android_control/data@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_gui_grpo_eval \
    trainer.n_gpus_per_node=8
