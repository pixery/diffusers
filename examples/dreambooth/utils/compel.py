import torch

from compel import Compel, ReturnedEmbeddingsType


def load_compels(pipe):
    compel_1 = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        truncate_long_prompts=False,
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=False,
    )
    compel_2 = Compel(
        tokenizer=pipe.tokenizer_2,
        text_encoder=pipe.text_encoder_2,
        truncate_long_prompts=False,
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=True,
    )
    compel_1.conditioning_provider.tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True
    compel_2.conditioning_provider.tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True
    return compel_1, compel_2


def _pad_to_same_length(*, cond_1, cond_2, compel_1, compel_2):
    conds_to_ensure_padded = [cond_1, cond_2]

    max_token_count = max([c.size(1) for c in conds_to_ensure_padded])

    padder_compel = compel_1 if cond_1.size(0) < cond_2.size(0) else compel_2
    empty_cond = padder_compel.build_conditioning_tensor("")
    if isinstance(empty_cond, tuple):
        empty_cond = empty_cond[0]  # discard pooled

    # if necessary, pad shorter tensors out with an emptystring tensor
    for i, c in enumerate(conds_to_ensure_padded):
        while c.size(1) < max_token_count:
            c = torch.cat([c, empty_cond], dim=1)
            conds_to_ensure_padded[i] = c

    cond_1, cond_2 = conds_to_ensure_padded
    return cond_1, cond_2


def get_prompt_embeds(*, compel_1, compel_2, prompt, neg_prompt=None, prompt_2=None, neg_prompt_2=None):
    if prompt_2 is None:
        prompt_2 = prompt
    if neg_prompt_2 is None:
        neg_prompt_2 = neg_prompt

    if neg_prompt is not None:
        cond_1, cond_neg_1 = compel_1([prompt, neg_prompt])
        (cond_2, cond_neg_2), (pooled, pooled_neg) = compel_2([prompt_2, neg_prompt_2])
        cond_1, cond_2 = _pad_to_same_length(
            cond_1=cond_1.unsqueeze(0),
            cond_2=cond_2.unsqueeze(0),
            compel_1=compel_1,
            compel_2=compel_2,
        )
        cond_neg_1, cond_neg_2 = _pad_to_same_length(
            cond_1=cond_neg_1.unsqueeze(0),
            cond_2=cond_neg_2.unsqueeze(0),
            compel_1=compel_1,
            compel_2=compel_2,
        )
        cond = torch.cat((cond_1, cond_2), dim=-1)
        cond_neg = torch.cat((cond_neg_1, cond_neg_2), dim=-1)
        pooled = pooled.unsqueeze(0)
        pooled_neg = pooled_neg.unsqueeze(0)
        return cond, pooled, cond_neg, pooled_neg
    else:
        cond_1 = compel_1(prompt)
        cond_2, pooled = compel_2(prompt_2)
        cond_1, cond_2 = _pad_to_same_length(
            cond_1=cond_1,
            cond_2=cond_2,
            compel_1=compel_1,
            compel_2=compel_2,
        )
        cond = torch.cat((cond_1, cond_2), dim=-1)
        return cond, pooled, None, None


def infer_compel(
    *,
    pipe,
    compel_1,
    compel_2,
    prompt,
    neg_prompt,
    prompt_2,
    neg_prompt_2,
    num_inference_steps,
    guidance_scale,
    generator,
):
    cond, pooled, cond_neg, pooled_neg = get_prompt_embeds(
        compel_1=compel_1,
        compel_2=compel_2,
        prompt=prompt,
        neg_prompt=neg_prompt,
        prompt_2=prompt_2,
        neg_prompt_2=neg_prompt_2,
    )

    image = pipe(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        prompt_embeds=cond,
        negative_prompt_embeds=cond_neg,
        pooled_prompt_embeds=pooled,
        negative_pooled_prompt_embeds=pooled_neg,
    ).images[0]

    return image


def main():
    from diffusers import DiffusionPipeline

    from .schedulers import get_diffusers_scheduler

    prompt = "portrait photo of man wearing an expensive white suit, white background, fit"
    neg_prompt = (
        "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, cropped, out of frame, worst quality, low"
        "quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn "
        "hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, "
        "cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra "
        "legs, fused fingers, too many fingers, long neck"
    )

    prompt_2 = (
        "man, a commercial photo portrait, clear edge definition, unique and one-of-a-kind pieces, Fujifilm "
        "X-T4, Sony FE 85mm f/1. 4 GM"
    )
    neg_prompt_2 = (
        "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, "
        "extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, "
        "bad anatomy, blurred, text, watermark, grainy"
    )

    scheduler = "DPM++ 2S a"
    device = torch.device("cuda", 1)
    num_inference_steps = 30
    guidance_scale = 7.0

    pipeline_seed = 1337

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16)
    scheduler = get_diffusers_scheduler(scheduler, pipeline=pipe)
    pipe.scheduler = scheduler
    pipe = pipe.to(torch_device=device)
    compel_1, compel_2 = load_compels(pipe)

    generator = torch.Generator(device=device)
    generator.manual_seed(pipeline_seed)

    image = infer_compel(
        pipe=pipe,
        compel_1=compel_1,
        compel_2=compel_2,
        prompt=prompt,
        neg_prompt=neg_prompt,
        prompt_2=prompt_2,
        neg_prompt_2=neg_prompt_2,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    image.save("/home/avcr/Desktop/ihsan/diffusers-pixery/diffusers/test.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
