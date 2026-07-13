#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute final voxel-wise ISC maps for the three matched participant groups.

The script reads:
  matched_group_isc/matched_average_subject_ids_en.txt
  matched_group_isc/matched_average_subject_ids_cn.txt
  matched_group_isc/matched_average_subject_ids_fr.txt

and writes one NIfTI ISC map per language under isc_maps/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np

import set_paths as spath
from match_group_isc import (
    LANGS,
    SubjectFiles,
    collect_subject_files,
    load_language_run_matrix,
    validate_mask,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--lang",
        type=str,
        default="all",
        choices=["all", "en", "cn", "fr"],
        help="language to process",
    )
    p.add_argument(
        "--home-folder",
        type=str,
        default=spath.home_folder,
        help="root folder containing lpp_en_resampled, lpp_cn_resampled, lpp_fr_resampled",
    )
    p.add_argument(
        "--ids-dir",
        type=str,
        default=spath.matched_group_isc_dir,
        help="directory containing matched_average_subject_ids_<lang>.txt",
    )
    p.add_argument(
        "--common-mask",
        type=str,
        default=spath.common_mask_path,
        help="common EN/CN/FR mask",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(spath.home_folder, "isc_maps"),
        help="directory where final ISC NIfTI maps are saved",
    )

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--n-runs", type=int, default=int(spath.n_runs))

    p.add_argument("--trim", type=int, default=10)
    p.add_argument("--high-pass", type=float, default=1 / 128)
    p.add_argument("--t-r", type=float, default=float(spath.t_r))
    p.add_argument("--smoothing-fwhm", type=float, default=8.0)
    p.add_argument(
        "--average",
        type=str,
        default="fisher",
        choices=["fisher", "raw"],
        help="average correlation maps directly or after Fisher z transform",
    )

    p.add_argument("--detrend", action="store_true", default=True)
    p.add_argument("--no-detrend", dest="detrend", action="store_false")

    p.add_argument("--standardize", action="store_true", default=True)
    p.add_argument("--no-standardize", dest="standardize", action="store_false")

    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="parallel jobs for preprocessing run files; keep low if memory is limited",
    )

    return p.parse_args()


def load_ids(ids_file: str) -> List[str]:
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"Matched ID file not found: {ids_file}")

    with open(ids_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError(f"No subject IDs found in {ids_file}")

    return ids


def select_subject_files(
    all_subjects: Sequence[SubjectFiles],
    matched_ids: Sequence[str],
) -> List[SubjectFiles]:
    by_id = {subject.subject_id: subject for subject in all_subjects}
    missing = [sid for sid in matched_ids if sid not in by_id]

    if missing:
        raise FileNotFoundError(f"Matched subjects not found in resampled data: {missing}")

    return [by_id[sid] for sid in matched_ids]


def make_split_indices(
    n_subjects: int,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_subjects % 2 != 0:
        raise ValueError(f"Split-half ISC requires an even number of subjects, got {n_subjects}")

    rng = np.random.default_rng(seed)
    half = n_subjects // 2
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_splits):
        order = rng.permutation(n_subjects)
        splits.append((order[:half], order[half:]))

    return splits


def voxelwise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")

    x0 = x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)
    y0 = y.astype(np.float64, copy=False) - y.mean(axis=0, keepdims=True)

    numerator = np.sum(x0 * y0, axis=0, dtype=np.float64)
    x_norm = np.sum(x0 * x0, axis=0, dtype=np.float64)
    y_norm = np.sum(y0 * y0, axis=0, dtype=np.float64)
    denominator = np.sqrt(x_norm * y_norm)

    corr = np.zeros(x.shape[1], dtype=np.float64)
    np.divide(numerator, denominator, out=corr, where=denominator > 0)

    return corr.astype(np.float32, copy=False)


def save_nifti_map(vector: np.ndarray, mask_path: str, out_path: str) -> None:
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata() > 0
    n_voxels = int(mask_data.sum())

    if vector.shape[0] != n_voxels:
        raise ValueError(
            f"ISC vector length does not match mask: {vector.shape[0]} vs {n_voxels}"
        )

    data = np.zeros(mask_data.shape, dtype=np.float32)
    data[mask_data] = vector.astype(np.float32, copy=False)

    out_img = nib.Nifti1Image(data, affine=mask_img.affine, header=mask_img.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, out_path)


def compute_language_isc_map(
    lang: str,
    lang_i: int,
    args: argparse.Namespace,
    mask_path: str,
    common_mask_voxels: int,
) -> Dict:
    ids_file = os.path.join(args.ids_dir, f"matched_average_subject_ids_{lang}.txt")
    matched_ids = load_ids(ids_file)

    all_subjects = collect_subject_files(
        home_folder=args.home_folder,
        lang=lang,
        n_runs=args.n_runs,
    )
    subjects = select_subject_files(all_subjects, matched_ids)

    split_seed = args.seed + 1000 * (lang_i + 1)
    splits = make_split_indices(
        n_subjects=len(subjects),
        n_splits=args.n_splits,
        seed=split_seed,
    )

    print(
        f"[INFO] {lang.upper()}: n={len(subjects)}, splits={args.n_splits}, "
        f"seed={args.seed}, split_seed={split_seed}",
        flush=True,
    )

    acc = np.zeros(common_mask_voxels, dtype=np.float64)
    n_maps = 0

    for run_idx in range(args.n_runs):
        t_run = time.time()

        print(f"[INFO] {lang.upper()} loading run {run_idx + 1}/{args.n_runs}...", flush=True)

        run_mat = load_language_run_matrix(
            subjects=subjects,
            run_idx=run_idx,
            mask_path=mask_path,
            trim=args.trim,
            detrend=args.detrend,
            standardize=args.standardize,
            high_pass=args.high_pass,
            t_r=args.t_r,
            smoothing_fwhm=args.smoothing_fwhm,
            n_jobs=args.n_jobs,
        )

        for idx_a, idx_b in splits:
            half_a = run_mat[idx_a].mean(axis=0)
            half_b = run_mat[idx_b].mean(axis=0)
            corr = voxelwise_corr(half_a, half_b)

            if args.average == "fisher":
                corr = np.clip(corr, -0.999999, 0.999999)
                acc += np.arctanh(corr)
            else:
                acc += corr

            n_maps += 1

        print(
            f"[INFO] {lang.upper()} run {run_idx + 1} done in {time.time() - t_run:.1f}s",
            flush=True,
        )

        del run_mat

    if args.average == "fisher":
        isc_map = np.tanh(acc / n_maps).astype(np.float32)
    else:
        isc_map = (acc / n_maps).astype(np.float32)

    out_name = f"isc_{lang}_matched_n{len(subjects)}_seed{args.seed}.nii.gz"
    out_path = os.path.join(args.out_dir, out_name)
    save_nifti_map(isc_map, mask_path, out_path)

    return {
        "lang": lang,
        "output_file": out_path,
        "ids_file": ids_file,
        "n_subjects": len(subjects),
        "subject_ids": matched_ids,
        "seed": args.seed,
        "split_seed": split_seed,
        "n_splits": args.n_splits,
        "n_runs": args.n_runs,
        "n_run_split_maps": n_maps,
        "average": args.average,
        "mean_isc_inside_mask": float(np.mean(isc_map)),
        "std_isc_inside_mask": float(np.std(isc_map)),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mask_path, common_mask_voxels = validate_mask(args.common_mask)
    langs = list(LANGS) if args.lang == "all" else [args.lang]

    summaries = []
    for lang in langs:
        lang_i = list(LANGS).index(lang)
        summaries.append(
            compute_language_isc_map(
                lang=lang,
                lang_i=lang_i,
                args=args,
                mask_path=mask_path,
                common_mask_voxels=common_mask_voxels,
            )
        )

    summary = {
        "method": "final_matched_group_voxelwise_split_half_isc_maps",
        "common_mask_path": mask_path,
        "common_mask_voxels": common_mask_voxels,
        "params": {
            "home_folder": args.home_folder,
            "ids_dir": args.ids_dir,
            "out_dir": args.out_dir,
            "seed": args.seed,
            "n_splits": args.n_splits,
            "n_runs": args.n_runs,
            "trim": args.trim,
            "high_pass": args.high_pass,
            "t_r": args.t_r,
            "smoothing_fwhm": args.smoothing_fwhm,
            "average": args.average,
            "detrend": args.detrend,
            "standardize": args.standardize,
            "n_jobs": args.n_jobs,
        },
        "maps": summaries,
    }

    summary_path = os.path.join(args.out_dir, "final_isc_maps_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Summary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
