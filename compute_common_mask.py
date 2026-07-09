#!/usr/bin/env python3

"""Compute symmetrized LPP EPI masks from resampled BOLD images.

This script is modified from `compute_mask.py` in
https://github.com/l-bg/llms_brain_lateralization.

The original script computed a symmetrized mask for one language. This version
keeps that behavior with `--lang en|cn|fr` and adds `--lang all` to compute one
common EN/CN/FR mask for the ISC-matched average-subject pipeline.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Sequence

import nibabel as nib
import numpy as np
from nilearn.image import swap_img_hemispheres
from nilearn.masking import compute_multi_epi_mask, intersect_masks

import set_paths as spath


LANGS = ("en", "cn", "fr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        choices=("all", *LANGS),
        help="mask to compute: all for one common EN/CN/FR mask, or one language",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output mask path; defaults to masks/mask_lpp_all.nii.gz or masks/mask_lpp_<lang>.nii.gz",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=None,
        help="optional unsymmetrized mask output path",
    )
    return parser.parse_args()


def default_output_paths(lang: str) -> tuple[str, str]:
    if lang == "all":
        return (
            spath.common_mask_path,
            os.path.join(spath.masks_root, "mask_lpp_all_raw.nii.gz"),
        )

    return (
        os.path.join(spath.masks_root, f"mask_lpp_{lang}.nii.gz"),
        os.path.join(spath.masks_root, f"mask_lpp_{lang}_raw.nii.gz"),
    )


def collect_images(langs: Sequence[str]) -> tuple[list[str], dict[str, int], dict[str, int]]:
    all_imgs: list[str] = []
    subject_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}

    for lang in langs:
        fmri_data_resampled = os.path.join(spath.home_folder, f"lpp_{lang}_resampled")
        subject_dirs = np.sort(
            glob.glob(os.path.join(fmri_data_resampled, f"sub-{lang.upper()}*"))
        )
        if subject_dirs.size == 0:
            raise FileNotFoundError(f"No resampled subjects found in {fmri_data_resampled}")

        subject_counts[lang] = int(subject_dirs.size)
        image_counts[lang] = 0

        for subject_dir in subject_dirs:
            run_imgs = sorted(glob.glob(os.path.join(subject_dir, "*_run*.nii.gz")))
            if not run_imgs:
                raise FileNotFoundError(f"No run images found in {subject_dir}")
            all_imgs.extend(run_imgs)
            image_counts[lang] += len(run_imgs)

    return all_imgs, subject_counts, image_counts


def main() -> None:
    args = parse_args()

    langs = LANGS if args.lang == "all" else (args.lang,)
    output_path, raw_output_path = default_output_paths(args.lang)
    output_path = args.output or output_path
    raw_output_path = args.raw_output or raw_output_path

    all_imgs, subject_counts, image_counts = collect_images(langs)
    print(f"Languages: {', '.join(lang.upper() for lang in langs)}")
    print(f"Subject counts: {subject_counts}")
    print(f"Image counts: {image_counts}")
    print(f"Total number of images: {len(all_imgs)}")

    raw_mask = compute_multi_epi_mask(all_imgs, threshold=args.threshold)
    sym_mask = intersect_masks(
        [raw_mask, swap_img_hemispheres(raw_mask)],
        threshold=1,
    )

    spath.make_dir(os.path.dirname(output_path))
    spath.make_dir(os.path.dirname(raw_output_path))
    nib.save(raw_mask, raw_output_path)
    nib.save(sym_mask, output_path)

    raw_voxels = int(np.asarray(raw_mask.dataobj).sum())
    sym_voxels = int(np.asarray(sym_mask.dataobj).sum())
    print(f"Saved raw mask to: {raw_output_path} ({raw_voxels} voxels)")
    print(f"Saved symmetrized mask to: {output_path} ({sym_voxels} voxels)")


if __name__ == "__main__":
    main()
