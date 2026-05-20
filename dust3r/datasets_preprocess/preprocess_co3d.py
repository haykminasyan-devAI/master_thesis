#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the CO3D dataset.
# Usage:
# python3 datasets_preprocess/preprocess_co3d.py --co3d_dir /path/to/co3d
# --------------------------------------------------------

import argparse
import random
import gzip
import json
import os
import os.path as osp

import torch
import PIL.Image
import numpy as np
import cv2

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import path_to_root  # noqa
import dust3r.datasets.utils.cropping as cropping  # noqa


CATEGORIES = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave",
    "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign",
    "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv",
    "umbrella", "vase", "wineglass",
]
CATEGORIES_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}  # for seeding

SINGLE_SEQUENCE_CATEGORIES = sorted(set(CATEGORIES) - set(["microwave", "stopsign", "tv"]))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument('--single_sequence_subset', default=False, action='store_true',
                        help="prepare the single_sequence_subset instead.")
    parser.add_argument("--output_dir", type=str, default="data/co3d_processed")
    parser.add_argument("--co3d_dir", type=str, required=True)
    parser.add_argument("--num_sequences_per_object", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_quality", type=float, default=0.5, help="Minimum viewpoint quality score.")

    parser.add_argument("--img_size", type=int, default=512,
                        help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
    parser.add_argument(
        "--fail_on_missing",
        action="store_true",
        help="Abort on missing RGB/depth/mask (default: skip missing frames for partial CO3D downloads)",
    )
    parser.add_argument(
        "--selected_sequences_json",
        type=str,
        default=None,
        help=(
            "Path to a JSON list of sequence names (e.g. raw CO3D <cat>_selected_8.json). "
            "Only those sequences are kept for this category (no random subsampling). "
            "Sequences that only exist in the official CO3D test split appear in selected_seqs_test.json; "
            "prepare_seqlevel_split_10cat8.py merges train+test when building the 8-seq split."
        ),
    )
    return parser


def convert_ndc_to_pinhole(focal_length, principal_point, image_size):
    focal_length = np.array(focal_length)
    principal_point = np.array(principal_point)
    image_size_wh = np.array([image_size[1], image_size[0]])
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    fx, fy = focal_length_px[0], focal_length_px[1]
    cx, cy = principal_point_px[0], principal_point_px[1]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    R = torch.from_numpy(R)[None, :, :]
    T = torch.from_numpy(T)[None, :]
    focal = torch.from_numpy(focal)[None, :]
    p0 = torch.from_numpy(p0)[None, :]
    image_size = torch.from_numpy(image_size)[None, :]

    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]


def get_set_list(category_dir, split, is_single_sequence_subset=False):
    listfiles = os.listdir(osp.join(category_dir, "set_lists"))
    if is_single_sequence_subset:
        # not all objects have manyview_dev
        subset_list_files = [f for f in listfiles if "manyview_dev" in f]
    else:
        subset_list_files = [f for f in listfiles if f"fewview_train" in f]

    sequences_all = []
    for subset_list_file in subset_list_files:
        with open(osp.join(category_dir, "set_lists", subset_list_file)) as f:
            subset_lists_data = json.load(f)
            sequences_all.extend(subset_lists_data[split])

    return sequences_all


def prepare_sequences(
    category,
    co3d_dir,
    output_dir,
    img_size,
    split,
    min_quality,
    max_num_sequences_per_object,
    seed,
    is_single_sequence_subset=False,
    fail_on_missing=False,
    restrict_sequence_names=None,
):
    random.seed(seed)
    category_dir = osp.join(co3d_dir, category)
    category_output_dir = osp.join(output_dir, category)
    sequences_all = get_set_list(category_dir, split, is_single_sequence_subset)
    sequences_numbers = sorted(set(seq_name for seq_name, _, _ in sequences_all))

    frame_file = osp.join(category_dir, "frame_annotations.jgz")
    sequence_file = osp.join(category_dir, "sequence_annotations.jgz")

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())
    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())

    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        frame_data_processed.setdefault(sequence_name, {})[f_data["frame_number"]] = f_data

    good_quality_sequences = set()
    for seq_data in sequence_data:
        # Use >= so min_quality=0.0 keeps score==0 (strict > would drop them).
        if float(seq_data["viewpoint_quality_score"]) >= float(min_quality):
            good_quality_sequences.add(seq_data["sequence_name"])

    sequences_numbers = [seq_name for seq_name in sequences_numbers if seq_name in good_quality_sequences]
    in_split_names = {x for x, _, _ in sequences_all}
    if restrict_sequence_names is not None:
        rs = sorted(set(restrict_sequence_names))
        sequences_numbers = [s for s in sequences_numbers if s in rs]
        only_other_split = [s for s in rs if s not in in_split_names]
        if only_other_split:
            print(
                f"  [{split}] {category}: {len(only_other_split)} sequence(s) only in the other CO3D split "
                f"(will appear after merging train+test): {only_other_split[:5]}{'...' if len(only_other_split) > 5 else ''}"
            )
        selected_sequences_numbers = sequences_numbers
    elif len(sequences_numbers) < max_num_sequences_per_object:
        selected_sequences_numbers = sequences_numbers
    else:
        selected_sequences_numbers = random.sample(sequences_numbers, max_num_sequences_per_object)

    if restrict_sequence_names is not None and len(selected_sequences_numbers) == 0:
        return {}

    selected_sequences_numbers_dict = {seq_name: [] for seq_name in selected_sequences_numbers}
    sequences_all = [(seq_name, frame_number, filepath)
                     for seq_name, frame_number, filepath in sequences_all
                     if seq_name in selected_sequences_numbers_dict]

    if not fail_on_missing:
        kept = []
        for seq_name, frame_number, filepath in sequences_all:
            if seq_name not in frame_data_processed or frame_number not in frame_data_processed[seq_name]:
                continue
            fd = frame_data_processed[seq_name][frame_number]
            mask_path = filepath.replace("images", "masks").replace(".jpg", ".png")
            image_path = os.path.join(co3d_dir, filepath)
            depth_path = os.path.join(co3d_dir, fd["depth"]["path"])
            mask_path_full = os.path.join(co3d_dir, mask_path)
            if (
                os.path.isfile(image_path)
                and os.path.isfile(depth_path)
                and os.path.isfile(mask_path_full)
            ):
                kept.append((seq_name, frame_number, filepath))
        sequences_all = kept
        if len(sequences_all) == 0:
            if restrict_sequence_names is not None:
                return {}
            raise RuntimeError(
                f"{category}: no frames left after skipping missing files. "
                "Rebuild set_lists from local frames or use full CO3D."
            )

    for seq_name, frame_number, filepath in tqdm(sequences_all):
        frame_idx = int(filepath.split('/')[-1][5:-4])
        mask_path = filepath.replace("images", "masks").replace(".jpg", ".png")
        try:
            frame_data = frame_data_processed[seq_name][frame_number]
        except KeyError:
            if fail_on_missing:
                raise
            continue
        focal_length = frame_data["viewpoint"]["focal_length"]
        principal_point = frame_data["viewpoint"]["principal_point"]
        image_size = frame_data["image"]["size"]
        K = convert_ndc_to_pinhole(focal_length, principal_point, image_size)
        R, tvec, camera_intrinsics = opencv_from_cameras_projection(np.array(frame_data["viewpoint"]["R"]),
                                                                    np.array(frame_data["viewpoint"]["T"]),
                                                                    np.array(focal_length),
                                                                    np.array(principal_point),
                                                                    np.array(image_size))

        depth_path = os.path.join(co3d_dir, frame_data["depth"]["path"])
        assert frame_data["depth"]["scale_adjustment"] == 1.0
        image_path = os.path.join(co3d_dir, filepath)
        mask_path_full = os.path.join(co3d_dir, mask_path)

        try:
            input_rgb_image = PIL.Image.open(image_path).convert('RGB')
            input_mask = plt.imread(mask_path_full)
        except (OSError, FileNotFoundError):
            if fail_on_missing:
                raise
            continue

        try:
            depth_pil = PIL.Image.open(depth_path)
        except (OSError, FileNotFoundError):
            if fail_on_missing:
                raise
            continue
        with depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            input_depthmap = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0])))
        depth_mask = np.stack((input_depthmap, input_mask), axis=-1)
        H, W = input_depthmap.shape

        camera_intrinsics = camera_intrinsics.numpy()
        cx, cy = camera_intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)

        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        input_rgb_image, depth_mask, input_camera_intrinsics = cropping.crop_image_depthmap(
            input_rgb_image, depth_mask, camera_intrinsics, crop_bbox)

        # try to set the lower dimension to img_size * 3/4 -> img_size=512 => 384
        scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
        output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
        if max(output_resolution) < img_size:
            # let's put the max dimension to img_size
            scale_final = (img_size / max(H, W)) + 1e-8
            output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)

        input_rgb_image, depth_mask, input_camera_intrinsics = cropping.rescale_image_depthmap(
            input_rgb_image, depth_mask, input_camera_intrinsics, output_resolution)
        input_depthmap = depth_mask[:, :, 0]
        input_mask = depth_mask[:, :, 1]

        # generate and adjust camera pose
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = tvec
        camera_pose = np.linalg.inv(camera_pose)

        # save crop images and depth, metadata
        save_img_path = os.path.join(output_dir, filepath)
        save_depth_path = os.path.join(output_dir, frame_data["depth"]["path"])
        save_mask_path = os.path.join(output_dir, mask_path)
        os.makedirs(os.path.split(save_img_path)[0], exist_ok=True)
        os.makedirs(os.path.split(save_depth_path)[0], exist_ok=True)
        os.makedirs(os.path.split(save_mask_path)[0], exist_ok=True)

        input_rgb_image.save(save_img_path)
        dmax = float(np.nanmax(input_depthmap))
        if not np.isfinite(dmax) or dmax <= 0:
            if fail_on_missing:
                raise ValueError(f"Invalid depth max for {image_path}")
            continue
        scaled_depth_map = (input_depthmap / dmax * 65535).astype(np.uint16)
        cv2.imwrite(save_depth_path, scaled_depth_map)
        cv2.imwrite(save_mask_path, (input_mask * 255).astype(np.uint8))

        save_meta_path = save_img_path.replace('jpg', 'npz')
        np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,
                 camera_pose=camera_pose, maximum_depth=dmax)

        selected_sequences_numbers_dict[seq_name].append(frame_idx)

    out = {k: v for k, v in selected_sequences_numbers_dict.items() if len(v) > 0}
    if not out:
        if restrict_sequence_names is not None:
            return {}
        raise RuntimeError(
            f"{category}: no sequences with successfully processed frames. "
            "Check downloads or rebuild set_lists from local frames only."
        )
    return out


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert args.co3d_dir != args.output_dir
    if args.category is None:
        if args.single_sequence_subset:
            categories = SINGLE_SEQUENCE_CATEGORIES
        else:
            categories = CATEGORIES
    else:
        categories = [args.category]
    os.makedirs(args.output_dir, exist_ok=True)

    restrict_list = None
    if args.selected_sequences_json:
        if args.category is None:
            raise ValueError("--selected_sequences_json requires --category (one category per command)")
        with open(args.selected_sequences_json, "r") as fid:
            restrict_list = json.load(fid)
        if not isinstance(restrict_list, list):
            raise ValueError("--selected_sequences_json must contain a JSON list of sequence name strings")

    for split in ['train', 'test']:
        selected_sequences_path = os.path.join(args.output_dir, f'selected_seqs_{split}.json')
        if os.path.isfile(selected_sequences_path):
            continue

        all_selected_sequences = {}
        for category in categories:
            category_output_dir = osp.join(args.output_dir, category)
            os.makedirs(category_output_dir, exist_ok=True)
            category_selected_sequences_path = os.path.join(category_output_dir, f'selected_seqs_{split}.json')
            if os.path.isfile(category_selected_sequences_path):
                with open(category_selected_sequences_path, 'r') as fid:
                    category_selected_sequences = json.load(fid)
            else:
                print(f"Processing {split} - category = {category}")
                category_selected_sequences = prepare_sequences(
                    category=category,
                    co3d_dir=args.co3d_dir,
                    output_dir=args.output_dir,
                    img_size=args.img_size,
                    split=split,
                    min_quality=args.min_quality,
                    max_num_sequences_per_object=args.num_sequences_per_object,
                    seed=args.seed + CATEGORIES_IDX[category],
                    is_single_sequence_subset=args.single_sequence_subset,
                    fail_on_missing=args.fail_on_missing,
                    restrict_sequence_names=restrict_list,
                )
                with open(category_selected_sequences_path, 'w') as file:
                    json.dump(category_selected_sequences, file)

            all_selected_sequences[category] = category_selected_sequences
        with open(selected_sequences_path, 'w') as file:
            json.dump(all_selected_sequences, file)
