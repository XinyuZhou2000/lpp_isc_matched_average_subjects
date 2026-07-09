#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute average-subject fMRI from matched participant ID txt files.

Input txt files (default):
  matched_group_isc/matched_average_subject_ids_en.txt
  matched_group_isc/matched_average_subject_ids_cn.txt
  matched_group_isc/matched_average_subject_ids_fr.txt

For each language, saves:
  lpp_averaged_subject/lpp_<lang>_average_subject_matched/average_subject_run-<run>.gz
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import joblib
import numpy as np
from tqdm import tqdm
from nilearn.input_data import NiftiMasker

import set_paths as spath
from set_paths import make_dir, standardize


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
        "--ids-dir",
        type=str,
        default=spath.matched_group_isc_dir,
        help="directory containing matched_average_subject_ids_<lang>.txt",
    )
    p.add_argument(
        "--out-suffix",
        type=str,
        default="average_subject_matched",
        help="output folder suffix: lpp_<lang>_<out-suffix>",
    )
    p.add_argument(
        "--mask-path",
        type=str,
        default=spath.common_mask_path,
        help="path to the common mask used for all languages",
    )
    p.add_argument("--trim", type=int, default=0, help="trim timepoints at both ends")
    p.add_argument("--high-pass", type=float, default=1 / 128)
    p.add_argument("--t-r", type=float, default=float(spath.t_r))
    p.add_argument("--detrend", action="store_true", default=True)
    p.add_argument("--no-detrend", dest="detrend", action="store_false")
    p.add_argument("--standardize", action="store_true", default=True)
    p.add_argument("--no-standardize", dest="standardize", action="store_false")
    return p.parse_args()


def load_ids(ids_file: str) -> List[str]:
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"IDs file not found: {ids_file}")
    with open(ids_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise ValueError(f"No IDs found in: {ids_file}")
    # preserve order but remove duplicates
    ids = list(dict.fromkeys(ids))
    return ids


def compute_one_language(lang: str, args: argparse.Namespace) -> Dict:
    ids_file = os.path.join(args.ids_dir, f"matched_average_subject_ids_{lang}.txt")
    subject_ids = load_ids(ids_file)

    fmri_data_resampled = os.path.join(spath.home_folder, f"lpp_{lang}_resampled")
    mask_path = args.mask_path
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    subject_dirs: List[str] = []
    for sid in subject_ids:
        sdir = os.path.join(fmri_data_resampled, sid)
        if not os.path.isdir(sdir):
            raise FileNotFoundError(f"Subject directory not found: {sdir}")
        subject_dirs.append(sdir)

    n_runs = int(spath.n_runs)
    masker = NiftiMasker(
        mask_img=mask_path,
        detrend=args.detrend,
        standardize=args.standardize,
        high_pass=args.high_pass,
        t_r=args.t_r,
    )
    masker.fit()

    run_sums = [None for _ in range(n_runs)]
    n_subjects = len(subject_dirs)

    for sdir in tqdm(subject_dirs, desc=f"[{lang.upper()}] subjects"):
        run_files = sorted(glob.glob(os.path.join(sdir, "*.nii.gz")))
        if len(run_files) < n_runs:
            raise ValueError(f"{os.path.basename(sdir)} has {len(run_files)} runs, expected >= {n_runs}")
        run_files = run_files[:n_runs]

        for run_idx, run_file in enumerate(run_files):
            X = masker.transform(run_file)  # (T, V)
            if args.trim > 0:
                if X.shape[0] <= 2 * args.trim:
                    raise ValueError(
                        f"Timepoints too short after trim for {run_file}: T={X.shape[0]}, trim={args.trim}"
                    )
                X = X[args.trim:-args.trim]
            if run_sums[run_idx] is None:
                run_sums[run_idx] = np.zeros_like(X, dtype=np.float64)
            run_sums[run_idx] += X

    out_dir = os.path.join(spath.output_root, f"lpp_{lang}_{args.out_suffix}")
    make_dir(out_dir)

    out_files: List[str] = []
    for run_idx in range(n_runs):
        avg = run_sums[run_idx] / n_subjects
        avg = standardize(avg, axis=0)
        out_file = os.path.join(out_dir, f"average_subject_run-{run_idx}.gz")
        joblib.dump(avg, out_file, compress=4)
        out_files.append(out_file)

    summary = {
        "lang": lang,
        "ids_file": ids_file,
        "n_subjects": n_subjects,
        "subject_ids": subject_ids,
        "mask_path": mask_path,
        "n_runs": n_runs,
        "output_dir": out_dir,
        "output_files": out_files,
        "trim": args.trim,
        "detrend": args.detrend,
        "standardize": args.standardize,
        "high_pass": args.high_pass,
        "t_r": args.t_r,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    langs = ["en", "cn", "fr"] if args.lang == "all" else [args.lang]

    all_summary = []
    for lang in langs:
        print(f"[INFO] Processing {lang.upper()} ...", flush=True)
        s = compute_one_language(lang, args)
        all_summary.append(s)
        print(f"[INFO] Done {lang.upper()} -> {s['output_dir']}", flush=True)

    top_summary = {
        "langs": langs,
        "ids_dir": args.ids_dir,
        "out_suffix": args.out_suffix,
        "details": all_summary,
    }
    out_top = os.path.join(
        spath.output_root,
        "outputs",
        "matched_group_isc",
        f"average_subject_from_ids_summary_{'_'.join(langs)}.json",
    )
    make_dir(os.path.dirname(out_top))
    with open(out_top, "w", encoding="utf-8") as f:
        json.dump(top_summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Summary saved: {out_top}", flush=True)


if __name__ == "__main__":
    main()
