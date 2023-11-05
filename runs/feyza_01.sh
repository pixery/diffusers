#!/usr/bin/env bash
set -euxo pipefail

python examples/dreambooth/train_dreambooth_lora_sdxl.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
--instance_data_dir="inputs/input-images/raw/Feyza" \
--class_data_dir="inputs/class-images/sdxl_base_1.0-default_vae_0.9-bf16-150_imgs-1024/woman" \
--instance_prompt="photo of ohwx woman" \
--class_prompt="photo of woman" \
--num_validation_images=4 \
--num_class_images=150 \
--output_dir="outputs/feyza_01" \
--seed=1337 \
--resolution=1024 \
--train_batch_size=2 \
--max_train_steps=2500 \
--checkpointing_steps=250 \
--gradient_accumulation_steps=2 \
--gradient_checkpointing \
--learning_rate=1e-5 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--mixed_precision="bf16" \
--prior_generation_precision="bf16" \
--rank=32 \
--validation_prompts_path="inputs/validation-prompts/validation_prompts-00.json" \
--validation_steps=250 \
--validation_min_steps=1 \
--checkpointing_min_steps=1
