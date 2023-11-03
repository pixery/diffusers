import json
from pathlib import Path
from typing import Literal, NamedTuple

from natsort import natsorted, ns


SCHEDULER_TYPE = Literal[
    "Euler",
    "Euler a",
    "DDIM",
    "PLMS",
    "DPM++ 2S a",
    "DPM++ 2S a Karras",
    "DPM++ SDE",
    "DPM++ SDE Karras",
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
]


class InferenceConfig(NamedTuple):
    prompt: str
    scheduler: SCHEDULER_TYPE
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str | None = None
    prompt_2: str | None = None
    negative_prompt_2: str | None = None
    ignore_variance: bool = True


TEST_CONFIGS = [
    # https://tinyurl.com/y4x5sa8x
    InferenceConfig(
        prompt="photo of ohwx man",
        scheduler="DPM++ 2S a",
        num_inference_steps=20,
        guidance_scale=7.0,
    ),
    # https://tinyurl.com/y4x5sa8x
    InferenceConfig(
        prompt="photo of ohwx man, 1 /4 headshot, from side, looking away, mouth closed, smiling",
        scheduler="DPM++ 2S a",
        num_inference_steps=20,
        guidance_scale=7.0,
    ),
    # https://tinyurl.com/y4x5sa8x
    InferenceConfig(
        prompt="breathtaking selfie ohwx man on a cliff, Fjords in background, award winning,professional",
        scheduler="DPM++ 2S a",
        num_inference_steps=20,
        guidance_scale=7.0,
    ),
    # # https://tinyurl.com/y4x5sa8x; prompt order as in https://tinyurl.com/d2yutk6z
    # InferenceConfig(
    #     prompt="ohwx man, breathtaking selfie, on a cliff, Fjords in background, award winning,professional",
    #     scheduler="DPM++ 2S a",
    #     num_inference_steps=20,
    #     guidance_scale=7.0,
    # ),
    # https://tinyurl.com/y4x5sa8x
    InferenceConfig(
        prompt="photo of ohwx man in Southern France, wearing a plain red t - shirt and darkblue shorts, Provence, lavender fields in background, hdr",
        scheduler="DPM++ 2S a",
        num_inference_steps=20,
        guidance_scale=7.0,
    ),
    # # https://tinyurl.com/y4x5sa8x; prompt order as in https://tinyurl.com/d2yutk6z
    # InferenceConfig(
    #     prompt="ohwx man, photo, in Southern France, wearing a plain red t - shirt and darkblue shorts, Provence, lavender fields in background, hdr",
    #     scheduler="DPM++ 2S a",
    #     num_inference_steps=20,
    #     guidance_scale=7.0,
    # ),
    # https://tinyurl.com/3rdkwfhf
    InferenceConfig(
        prompt="portrait photo of (ohwx man)+ wearing an expensive white suit, white background, fit",
        negative_prompt="(semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)1.4, cropped, out of "
        "frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, "
        "mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, "
        "bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, "
        "missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        scheduler="DPM++ 2S a",
        num_inference_steps=30,
        guidance_scale=7.0,  # reduce it: https://github.com/huggingface/diffusers/issues/2431#issuecomment-1500607152
    ),
    # # https://tinyurl.com/3rdkwfhf; prompt order as in https://tinyurl.com/d2yutk6z
    # InferenceConfig(
    #     prompt="ohwx man, portrait photo, wearing an expensive white suit, white background, fit",
    #     negative_prompt="(semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)1.4, cropped, out of "
    #     "frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, "
    #     "mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, "
    #     "bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, "
    #     "missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
    #     scheduler="DPM++ 2S a",
    #     num_inference_steps=30,
    #     guidance_scale=7.0,  # reduce it: https://github.com/huggingface/diffusers/issues/2431#issuecomment-1500607152
    # ),
    # https://huggingface.co/minimaxir/sdxl-wrong-lora
    InferenceConfig(
        prompt="realistic ohwx man blogging at a computer workstation, hyperrealistic award-winning photo for vanity "
        "fair",
        negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, "
        "mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, "
        "blurry, bad art, bad anatomy, blurred, text, watermark, grainy",  # https://tinyurl.com/36sr74bp
        # scheduler="Euler",
        # num_inference_steps=50,
        # guidance_scale=13.0,
        scheduler="DPM++ 2S a",
        num_inference_steps=30,
        guidance_scale=13.0,
    ),
    # # https://huggingface.co/minimaxir/sdxl-wrong-lora; prompt order as in https://tinyurl.com/d2yutk6z
    # InferenceConfig(
    #     prompt="ohwx man, realistic, blogging at a computer workstation, hyperrealistic award-winning photo for "
    #     "vanity fair",
    #     negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, "
    #     "mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, "
    #     "blurry, bad art, bad anatomy, blurred, text, watermark, grainy",  # https://tinyurl.com/36sr74bp
    #     scheduler="Euler",
    #     num_inference_steps=50,
    #     guidance_scale=13.0,
    # ),
    # https://tinyurl.com/n5psf59c
    InferenceConfig(
        prompt="a commercial photo portrait of ohwx man, clear edge definition, unique and one-of-a-kind pieces, "
        "Fujifilm X-T4, Sony FE 85mm f/1. 4 GM",
        negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, "
        "mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, "
        "blurry, bad art, bad anatomy, blurred, text, watermark, grainy",  # https://tinyurl.com/36sr74bp
        # scheduler="Euler",
        # num_inference_steps=100,
        # guidance_scale=5.0,
        scheduler="DPM++ 2S a",
        num_inference_steps=30,
        guidance_scale=5.0,
    ),
    # # https://tinyurl.com/n5psf59c; prompt order as in https://tinyurl.com/d2yutk6z
    # InferenceConfig(
    #     prompt="ohwx man, a commercial photo portrait, clear edge definition, unique and one-of-a-kind pieces, "
    #     "Fujifilm X-T4, Sony FE 85mm f/1. 4 GM",
    #     negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, "
    #     "mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, "
    #     "blurry, bad art, bad anatomy, blurred, text, watermark, grainy",  # https://tinyurl.com/36sr74bp
    #     scheduler="Euler",
    #     num_inference_steps=100,
    #     guidance_scale=5.0,
    # ),
]

SAVE_DIR = Path("/home/avcr/Desktop/ihsan/diffusers-pixery/diffusers/inputs/validation-prompts")


def get_save_name():
    current_files = [p.stem for p in SAVE_DIR.iterdir()]
    current_files = natsorted(current_files, alg=ns.PATH)
    most_recent_file_id = max([int(f.split("-")[1]) for f in current_files] or [-1])
    current_file_id = most_recent_file_id + 1
    save_name = f"validation_prompts-{current_file_id:02d}.json"
    return save_name


def dump_configs():
    configs = [c._asdict() for c in TEST_CONFIGS]
    save_name = get_save_name()
    save_path = SAVE_DIR / save_name
    with open(save_path, "w") as f:
        json.dump(configs, f, indent=2)


def main():
    dump_configs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
