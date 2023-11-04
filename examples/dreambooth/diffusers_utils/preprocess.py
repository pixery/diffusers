import shutil
from itertools import chain
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F

import mediapipe as mp
from natsort import natsorted, ns
from PIL import Image, ImageFilter, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Swin2SRForImageSuperResolution,
    Swin2SRImageProcessor,
)


register_heif_opener()

IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.heic"]
IMAGE_EXTENSIONS += [e.upper() for e in IMAGE_EXTENSIONS]

DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRETRAINED_DIR = Path("/home/avcr/Desktop/ihsan/diffusers-pixery/diffusers/pretrained")

SEGFORMER_PATH = PRETRAINED_DIR / "nvidia/segformer-b5-finetuned-ade-640-640"
SWIN2SR_PATH = PRETRAINED_DIR / "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"


def get_face_bboxes(images):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)

    face_bboxes = []
    for image in tqdm(images):
        image_np = np.array(image)
        # Perform face detection
        results_detection = face_detection.process(image_np)
        ih, iw, *_ = image_np.shape
        if detections := results_detection.detections:
            if len(detections) > 1:
                raise RuntimeError
            bbox_rel = detections[0].location_data.relative_bounding_box
            bbox_xywh = (
                int(bbox_rel.xmin * iw),
                int(bbox_rel.ymin * ih),
                int(bbox_rel.width * iw),
                int(bbox_rel.height * ih),
            )
            # make sure bbox is within image
            bbox_xywh = (
                max(0, bbox_xywh[0]),
                max(0, bbox_xywh[1]),
                min(iw - bbox_xywh[0], bbox_xywh[2]),
                min(ih - bbox_xywh[1], bbox_xywh[3]),
            )
            bbox_lurl = (
                bbox_xywh[0],
                bbox_xywh[1],
                bbox_xywh[0] + bbox_xywh[2],
                bbox_xywh[1] + bbox_xywh[3],
            )
            face_bboxes.append(bbox_lurl)
        else:
            face_bboxes.append(None)

    return face_bboxes


@torch.no_grad()
@torch.cuda.amp.autocast()
def segment(images, *, device=DEFAULT_DEVICE):
    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_PATH)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_PATH)
    model = model.to(device=device)

    masks = []
    for image in images:
        inputs = processor(images=image, return_tensors="pt").to(device=device)
        outputs = model(**inputs)
        logits = outputs.logits

        probs = F.softmax(logits, dim=1)
        probs_person = probs[:, [12], ...]
        probs_person = F.interpolate(
            probs_person, size=image.size[::-1], mode="bilinear", antialias=True  # (height, width)
        ).squeeze(dim=(0, 1))

        mask_person = (probs_person * 255).byte().cpu().numpy()
        mask_person = Image.fromarray(mask_person)
        masks.append(mask_person)

    return masks


def get_surrounding_box_for_mask(mask, pad_percentage=5.0):
    w, h = mask.size
    mask = np.array(mask)

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    crop_width = w - (crop_right + crop_left)
    crop_height = h - (crop_bottom + crop_top)
    pad_x = crop_width * (2 * pad_percentage / 100)
    pad_y = crop_height * (2 * pad_percentage / 100)
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top

    return (
        int(max(crop_left - pad_left, 0)),
        int(max(crop_top - pad_top, 0)),
        int(min(w - crop_right + pad_right, w)),
        int(min(h - crop_bottom + pad_bottom, h)),
    )


def expand_face_bboxes(bboxes, masks, pad_percentage=5.0):
    boxes_expanded = []
    for mask, box in zip(masks, bboxes):
        if box is not None:
            box_is_for_face = True

            box_left, box_top, box_right, box_bottom = box
            w, h = mask.size
            h_box = box_bottom - box_top
            w_box = box_right - box_left

            mask = np.array(mask)
            mask_top = 0
            for i in range(h):
                if not (mask[i] == 0).all():
                    break
                mask_top += 1
            top_margin = box_top - mask_top
            expand_factor = top_margin / h_box
            left_margin = round(expand_factor * w_box)

            margin_y = round(top_margin * (1 + pad_percentage / 100))
            margin_x = round(left_margin * (1 + pad_percentage / 100))

            box_expanded = (
                int(max(box_left - margin_x, 0)),
                int(max(box_top - margin_y, 0)),
                int(min(box_right + margin_x, w)),
                int(min(box_bottom + margin_y, h)),
            )
            boxes_expanded.append((box_expanded, box_is_for_face))
        else:
            box_is_for_face = False
            surrounding_box = get_surrounding_box_for_mask(mask)
            boxes_expanded.append((surrounding_box, box_is_for_face))

    return boxes_expanded


def get_face_masks_and_expanded_face_bboxes(images, face_bboxes, blur_amount=0.0, bias=50.0):
    masks_full = segment(images)
    boxes_expanded = expand_face_bboxes(face_bboxes, masks_full)

    masks_face = []
    for mask_full, face_bbox_and_status in tqdm(zip(masks_full, boxes_expanded)):
        face_bbox, box_is_for_face = face_bbox_and_status

        if box_is_for_face:
            mask_np = np.array(mask_full)
            iw, ih = mask_full.size

            bbox_zero_mask = np.ones((ih, iw), dtype=bool)
            bbox_zero_mask[face_bbox[1] : face_bbox[3], face_bbox[0] : face_bbox[2]] = False
            mask_np[bbox_zero_mask] = 0
            mask_face = Image.fromarray(mask_np)

            # Apply blur to the mask
            if blur_amount > 0:
                mask_face = mask_face.filter(ImageFilter.GaussianBlur(blur_amount))

            # Apply bias to the mask
            if bias > 0:
                mask_face = np.array(mask_face)
                mask_face = mask_face + bias * np.ones(mask_face.shape, dtype=mask_face.dtype)
                mask_face = np.clip(mask_face, 0, 255)
                mask_face = Image.fromarray(mask_face)

            # Convert mask to 'L' mode (grayscale) before saving
            mask_face = mask_face.convert("L")

            masks_face.append(mask_face)
        else:
            mask_np = np.array(mask_full)

            # Apply blur to the mask
            if blur_amount > 0:
                mask_full = mask_full.filter(ImageFilter.GaussianBlur(blur_amount))

            # Apply bias to the mask
            if bias > 0:
                mask_full = np.array(mask_full)
                mask_full = mask_full + bias * np.ones(mask_full.shape, dtype=mask_full.dtype)
                mask_full = np.clip(mask_full, 0, 255)
                mask_full = Image.fromarray(mask_full)

            # Convert mask to 'L' mode (grayscale) before saving
            mask_full = mask_full.convert("L")

            masks_face.append(mask_full)

    return masks_face, boxes_expanded


def center_of_mass(mask):
    """Returns the center of mass of the mask."""
    x, y = np.meshgrid(np.arange(mask.size[0]), np.arange(mask.size[1]))
    mask_np = np.array(mask) + 0.01
    x_ = x * mask_np
    y_ = y * mask_np

    x = np.sum(x_) / np.sum(mask_np)
    y = np.sum(y_) / np.sum(mask_np)

    return x, y


def get_box_for_trim_to_square(image, com):
    cx, cy = com
    width, height = image.size
    if width > height:
        left_possible = max(cx - height / 2, 0)
        left = min(left_possible, width - height)
        right = left + height
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top_possible = max(cy - width / 2, 0)
        top = min(top_possible, height - width)
        bottom = top + width
    trim_box = (left, top, right, bottom)
    return trim_box


def square_crop_to_face(image, face_bbox, expand_factor=1.0):
    iw, ih = image.size
    assert iw == ih

    box, box_is_for_face = face_bbox
    if not box_is_for_face:
        return image

    x1_face, y1_face, x2_face, y2_face = box

    center_x = (x1_face + x2_face) // 2
    center_y = (y1_face + y2_face) // 2

    square_size = int(max(x2_face - x1_face, y2_face - y1_face) * expand_factor)

    x1 = max(center_x - square_size // 2, 0)
    y1 = max(center_y - square_size // 2, 0)
    x2 = min(x1 + square_size, iw)
    y2 = min(y1 + square_size, ih)
    if y2 - y1 > x2 - x1:
        y1 = max(center_y - (x2 - x1) // 2, 0)
        y2 = min(y1 + (x2 - x1), ih)
    elif x2 - x1 > y2 - y1:
        x1 = max(center_x - (y2 - y1) // 2, 0)
        x2 = min(x1 + (y2 - y1), iw)

    crop_box = (x1, y1, x2, y2)
    image_cropped = image.crop(crop_box)

    return image_cropped


@torch.no_grad()
@torch.cuda.amp.autocast()
def swin_ir_sr(images, target_size=None, device=DEFAULT_DEVICE):
    """Upscales images using SwinIR.

    Returns a list of PIL images.

    If the image is already larger than the target size, it will not be upscaled and will be returned as is.
    """
    model = Swin2SRForImageSuperResolution.from_pretrained(SWIN2SR_PATH).to(device)
    processor = Swin2SRImageProcessor()

    out_images = []

    for image in tqdm(images):
        ori_w, ori_h = image.size
        if target_size is not None:
            if ori_w >= target_size[0] and ori_h >= target_size[1]:
                out_images.append(image)
                continue

        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)
        output = Image.fromarray(output)

        out_images.append(output)

    return out_images


def get_save_name(input_dir, target_size, trim_based_on_face, apply_face_crop, face_crop_expand_factor):
    character_id = input_dir.stem
    face_trim_status = f'{"do" if trim_based_on_face else "no"}facetrim'
    face_crop_status = f'{"do" if apply_face_crop else "no"}facecrop_{face_crop_expand_factor:.2f}'
    resolution_status = f"res_{target_size}"
    return "-".join([character_id, face_trim_status, face_crop_status, resolution_status])


def preprocess(
    input_dir,
    output_dir=None,
    target_size=1024,
    trim_based_on_face=True,
    apply_face_crop=True,
    face_crop_expand_factor=1.5,
    force_reprocess=False,
):
    """Loads images, saves masks.

    Loads images from the given files, generates masks for them, and saves the masks and upscaled images to output dir.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        input_root_dir = input_dir.parents[1]
        output_root_dir = input_root_dir / "processed"
        save_name = get_save_name(input_dir, target_size, trim_based_on_face, apply_face_crop, face_crop_expand_factor)
        output_dir = output_root_dir / save_name
    if output_dir.is_dir():
        if force_reprocess:
            shutil.rmtree(output_dir)
        else:
            return output_dir
    output_dir.mkdir(parents=True)

    # load images
    # get all the .png .jpg .heic in the directory
    file_paths = chain.from_iterable(input_dir.glob(p) for p in IMAGE_EXTENSIONS)
    file_paths = natsorted(file_paths, alg=ns.PATH)
    if len(file_paths) == 0:
        raise Exception(
            f"No files found in {str(input_dir)!r}. "
            f"Either {str(input_dir)!r} is not a directory or it does not contain any .png, .jpg/jpeg, or .heic files."
        )
    print("Image files:")
    pprint([str(p) for p in file_paths])
    images = [Image.open(path) for path in file_paths]
    images = [ImageOps.exif_transpose(im) for im in images]
    images = [im.convert("RGB") for im in images]

    print(f"Generating {len(images)} masks...")
    face_bboxes = get_face_bboxes(images)
    seg_masks, face_bboxes = get_face_masks_and_expanded_face_bboxes(images, face_bboxes)

    # based on the center of mass, trim the image to a square (chop sides off rectangular images, making them squares)
    if trim_based_on_face:
        coms = [center_of_mass(mask) for mask in seg_masks]
    else:
        coms = [(image.size[0] / 2, image.size[1] / 2) for image in images]
    trim_boxes = [get_box_for_trim_to_square(image, com) for image, com in zip(images, coms)]
    # adjust trim_boxes such that they also cover the face boxes
    trim_boxes_adjusted = []
    for ((face_left, face_upper, *_), _), (trim_left, trim_upper, trim_right, trim_lower) in zip(
        face_bboxes, trim_boxes
    ):
        delta_x = max(trim_left - face_left, 0)
        delta_y = max(trim_upper - face_upper, 0)
        trim_left = trim_left - delta_x
        trim_upper = trim_upper - delta_y
        trim_right = trim_right - delta_x
        trim_lower = trim_lower - delta_y
        trim_box_adjusted = (trim_left, trim_upper, trim_right, trim_lower)
        trim_boxes_adjusted.append(trim_box_adjusted)

    images = [image.crop(box) for image, box in zip(images, trim_boxes_adjusted)]
    seg_masks = [mask.crop(box) for mask, box in zip(seg_masks, trim_boxes_adjusted)]

    # adjust face bboxes for the post-trim coordinates
    face_bboxes = [
        (
            (face_left - trim_left, face_upper - trim_upper, face_right - trim_left, face_lower - trim_upper),
            face_status,
        )
        for ((face_left, face_upper, face_right, face_lower), face_status), (
            trim_left,
            trim_upper,
            *_,
        ) in zip(face_bboxes, trim_boxes_adjusted)
    ]

    if apply_face_crop:
        images = [
            square_crop_to_face(image, face_bbox, face_crop_expand_factor)
            for image, face_bbox in zip(images, face_bboxes)
        ]
        seg_masks = [
            square_crop_to_face(mask, face_bbox, face_crop_expand_factor)
            for mask, face_bbox in zip(seg_masks, face_bboxes)
        ]

    # resize images
    print(f"Upscaling {len(images)} images...")
    images = swin_ir_sr(images, target_size=(target_size, target_size))
    images = [image.resize((target_size, target_size), Image.Resampling.LANCZOS) for image in images]

    # resize masks
    seg_masks = [mask.resize((target_size, target_size), Image.Resampling.LANCZOS) for mask in seg_masks]

    # iterate through the images and masks
    for idx, (image, mask) in enumerate(zip(images, seg_masks)):
        image_name = f"{idx}.src.png"
        mask_file = f"{idx}.mask.png"

        # save the image and mask files
        image.save(output_dir / image_name)
        mask.save(output_dir / mask_file)

    return output_dir


def main():
    input_dir = Path("/home/avcr/Desktop/ihsan/diffusers-pixery/diffusers/inputs/input-images/raw/Mobici")
    output_dir = preprocess(input_dir, force_reprocess=True)
    print(f"Processed images saved to {str(output_dir)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
