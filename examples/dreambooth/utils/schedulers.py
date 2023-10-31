import inspect
import re
from functools import cache

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)


DIFFUSERS_SCHEDULER_CANDIDATES_BY_A1111_NAME = {
    "Euler": [
        (
            EulerDiscreteScheduler,
            {},
        ),
    ],
    "Euler a": [
        (
            EulerAncestralDiscreteScheduler,
            {},
        ),
    ],
    "DDIM": [
        (
            DDIMScheduler,
            {},
        ),
    ],
    "PLMS": [
        (
            PNDMScheduler,
            {
                "skip_prk_steps": True,
            },
        ),
    ],
    "DPM++ 2S a": [
        # ! Not implemented, and not planned
        (
            NotImplemented,
            {},
        ),
        #
        # * Suggested alternative: (https://github.com/huggingface/diffusers/issues/4167)
        # not released yet, however; pending https://github.com/huggingface/diffusers/pull/4251
        (
            DPMSolverSinglestepScheduler,
            {
                "algorithm_type": "sde-dpmsolver++",
            },
        ),
        #
        # * Suggested second alternative:
        # (https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview#schedulers)
        (
            DPMSolverSinglestepScheduler,
            {},
        ),
    ],
    "DPM++ 2S a Karras": [
        # ! Not implemented, and not planned
        (
            NotImplemented,
            {},
        ),
        #
        # * Suggested alternative: (https://github.com/huggingface/diffusers/issues/4167)
        # not released yet, however; pending https://github.com/huggingface/diffusers/pull/4251
        (
            DPMSolverSinglestepScheduler,
            {
                "algorithm_type": "sde-dpmsolver++",
                "use_karras_sigmas": True,
            },
        ),
        #
        # * Suggested second alternative:
        # (https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview#schedulers)
        (
            DPMSolverSinglestepScheduler,
            {
                "use_karras_sigmas": True,
            },
        ),
    ],
    "DPM++ SDE": [
        # * Not released yet; pending https://github.com/huggingface/diffusers/pull/4251
        (
            DPMSolverSinglestepScheduler,
            {
                "algorithm_type": "sde-dpmsolver++",
            },
        ),
        #
        # * Suggested alternative: (https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview#schedulers)
        (
            DPMSolverSinglestepScheduler,
            {},
        ),
    ],
    "DPM++ SDE Karras": [
        # * Not released yet; pending https://github.com/huggingface/diffusers/pull/4251
        (
            DPMSolverSinglestepScheduler,
            {
                "algorithm_type": "sde-dpmsolver++",
                "use_karras_sigmas": True,
            },
        ),
        #
        # * Suggested alternative: (https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview#schedulers)
        (
            DPMSolverSinglestepScheduler,
            {
                "use_karras_sigmas": True,
            },
        ),
    ],
    "DPM++ 2M": [
        (
            DPMSolverMultistepScheduler,
            {},
        ),
    ],
    "DPM++ 2M Karras": [
        (
            DPMSolverMultistepScheduler,
            {
                "use_karras_sigmas": True,
            },
        ),
    ],
    "DPM++ 2M SDE": [
        (
            DPMSolverMultistepScheduler,
            {
                "algorithm_type": "sde-dpmsolver++",
            },
        ),
    ],
    "DPM++ 2M SDE Karras": [
        (
            DPMSolverMultistepScheduler,
            {
                "algorithm_type": "sde-dpmsolver++",
                "use_karras_sigmas": True,
            },
        ),
    ],
}


def get_description(docstring, param_name):
    """By GPT-4.

    From: https://chat.openai.com/share/932fcb31-3fc5-48a9-b8f2-5db00717f5be.
    """
    pattern = r"{param}\s+\([^)]+\)\s*:([\s\S]*?)(?=\n\s{{4}}\w+\s+\([^)]+\)|$)".format(param=param_name)
    match = re.search(pattern, docstring)
    if match:
        return match.group(1).strip()
    else:
        return None


@cache
def get_diffusers_scheduler_class(a1111_name):
    for sch_class, params in DIFFUSERS_SCHEDULER_CANDIDATES_BY_A1111_NAME[a1111_name]:
        if sch_class is NotImplemented:
            print(f"{a1111_name!r} is not implemented in Diffusers; using a similar alternative ...")
            continue
        docstring = inspect.cleandoc(sch_class.__doc__)
        kwargs = ", ".join(f"{n}={v!r}" for n, v in params.items())
        for param_name, param_value in params.items():
            if not isinstance(param_value, str):
                continue
            description = get_description(docstring, param_name)
            if param_value not in description:
                print(
                    f"{a1111_name!r}: {sch_class.__name__}({kwargs}) "
                    f"is not implemented in Diffusers (illegal argument: {param_name}={param_value!r}); "
                    "using a similar alternative ..."
                )
                break
        else:
            print(f"{a1111_name!r}: Using {sch_class.__name__}({kwargs})")
            return sch_class, params


def scheduler_params_ignore_variance(params, *, pipeline):
    # See train_dreambooth_lora_sdxl.py
    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        params["variance_type"] = variance_type
    return params


def get_diffusers_scheduler(a1111_name, *, pipeline, ignore_variance=True):
    sch_class, params = get_diffusers_scheduler_class(a1111_name)
    if ignore_variance:
        params = scheduler_params_ignore_variance(params, pipeline=pipeline)
    try:
        scheduler = sch_class.from_config(pipeline.scheduler.config, **params)
    except NotImplementedError as e:
        kwargs = ", ".join(f"{n}={v!r}" for n, v in params.items())
        for param_name, param_value in list(params.items()):
            if param_value in str(e):
                print(
                    f"{a1111_name!r}: {sch_class.__name__}({kwargs}) "
                    f"is not implemented in Diffusers (illegal argument: {param_name}={param_value!r})"
                )
                del params[param_name]
        print("Using a similar alternative ...")
        kwargs = ", ".join(f"{n}={v!r}" for n, v in params.items())
        print(f"{a1111_name!r}: Using {sch_class.__name__}({kwargs})")
        scheduler = sch_class.from_config(pipeline.scheduler.config, **params)
    return scheduler


def main():
    a1111_name = "DPM++ 2S a Karras"
    _ = get_diffusers_scheduler_class(a1111_name)


if __name__ == "__main__":
    raise SystemExit(main())
