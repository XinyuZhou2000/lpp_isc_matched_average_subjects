#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert joblib-compressed average-subject arrays to NIfTI files.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask",
        type=str,
        default=str(base_dir / "masks" / "mask_lpp_all.nii.gz"),
        help="Path to the mask used to map voxel values back to 3D space.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[
            str(base_dir / "lpp_averaged_subject" / "lpp_cn_average_subject_matched"),
            str(base_dir / "lpp_averaged_subject" / "lpp_en_average_subject_matched"),
            str(base_dir / "lpp_averaged_subject" / "lpp_fr_average_subject_matched"),
        ],
        help="Folders containing average_subject_run-*.gz files.",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=2.0,
        help="Repetition time in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mask_path = Path(args.mask)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask_img = nib.load(str(mask_path))
    mask = np.asanyarray(mask_img.dataobj) > 0
    n_voxels = int(mask.sum())

    print(f"Mask: {mask_path}")
    print(f"Mask shape: {mask_img.shape}, nonzero voxels: {n_voxels}")

    for folder in args.folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"[SKIP] Missing folder: {folder_path}")
            continue

        gz_files = sorted(
            p for p in folder_path.glob("average_subject_run-*.gz")
            if not p.name.endswith(".nii.gz")
        )
        if not gz_files:
            print(f"[SKIP] No average_subject_run-*.gz files in {folder_path}")
            continue

        print(f"[PROCESS] {folder_path}")

        for gz_path in gz_files:
            arr = joblib.load(gz_path)
            arr = np.asarray(arr)

            if arr.ndim != 2:
                raise ValueError(
                    f"{gz_path} should be a 2D array, got shape {arr.shape}"
                )

            if arr.shape[1] != n_voxels:
                raise ValueError(
                    f"{gz_path} has {arr.shape[1]} voxels, "
                    f"but mask has {n_voxels}"
                )

            if not np.isfinite(arr).all():
                print(f"  [WARNING] Non-finite values found in {gz_path}")

            n_timepoints = arr.shape[0]

            data4d = np.zeros(mask.shape + (n_timepoints,), dtype=np.float32)
            data4d[mask, :] = arr.T.astype(np.float32)

            header = mask_img.header.copy()
            header.set_data_shape(data4d.shape)
            header.set_data_dtype(np.float32)
            header.set_zooms(mask_img.header.get_zooms()[:3] + (args.tr,))
            header.set_xyzt_units("mm", "sec")

            out_path = gz_path.with_suffix(".nii.gz")

            out_img = nib.Nifti1Image(data4d, mask_img.affine, header)
            nib.save(out_img, str(out_path))

            print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
