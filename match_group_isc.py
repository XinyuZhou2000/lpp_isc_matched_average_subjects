#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Match EN/CN/FR 28-subject groups by closest voxel-wise group-ISC scores.

For each candidate group, the script computes repeated split-half ISC. Within
each split, subjects are averaged within each half, Pearson correlation is
computed separately for each voxel, and voxel-wise correlations are averaged
across voxels, splits, and runs.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

try:
    from nilearn.maskers import NiftiMasker
except ImportError:
    from nilearn.input_data import NiftiMasker

import set_paths as spath


LANGS = ("en", "cn", "fr")

DEFAULT_HOME_FOLDER = spath.home_folder
DEFAULT_COMMON_MASK = spath.common_mask_path
DEFAULT_N_RUNS = int(spath.n_runs)
DEFAULT_TR = float(spath.t_r)


@dataclass
class SubjectFiles:
    subject_id: str
    run_files: List[str]


def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--home-folder",
        type=str,
        default=DEFAULT_HOME_FOLDER,
        help="root folder containing lpp_en_resampled, lpp_cn_resampled, lpp_fr_resampled",
    )

    p.add_argument(
        "--common-mask",
        type=str,
        default=DEFAULT_COMMON_MASK,
        help="precomputed common mask; this script will not recompute it",
    )

    p.add_argument("--group-size", type=int, default=28)
    p.add_argument("--n-candidates", type=int, default=50)
    p.add_argument("--n-splits", type=int, default=10)

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)

    p.add_argument("--trim", type=int, default=10)
    p.add_argument("--high-pass", type=float, default=1 / 128)
    p.add_argument("--t-r", type=float, default=DEFAULT_TR)
    p.add_argument("--smoothing-fwhm", type=float, default=8.0)

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

    p.add_argument(
        "--out-dir",
        type=str,
        default=spath.matched_group_isc_dir,
    )

    return p.parse_args()


def subject_numeric_id(subject_id: str, lang: str) -> int:
    pat = re.compile(rf"sub-{lang.upper()}(\d+)$")
    m = pat.match(subject_id)
    if not m:
        raise ValueError(f"Cannot parse numeric id from {subject_id}")
    return int(m.group(1))


def validate_mask(mask_path: str) -> Tuple[str, int]:
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Common mask not found: {mask_path}")

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata() > 0

    if mask_data.ndim != 3:
        raise ValueError(f"Common mask must be 3D, got shape={mask_data.shape}")

    return mask_path, int(mask_data.sum())


def collect_subject_files(
    home_folder: str,
    lang: str,
    n_runs: int,
) -> List[SubjectFiles]:
    root = os.path.join(home_folder, f"lpp_{lang}_resampled")
    subject_dirs = glob.glob(os.path.join(root, f"sub-{lang.upper()}*"))

    if not subject_dirs:
        raise FileNotFoundError(f"No subjects found in {root}")

    subject_dirs = sorted(
        subject_dirs,
        key=lambda d: subject_numeric_id(os.path.basename(d), lang),
    )

    subjects: List[SubjectFiles] = []

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)

        run_files = sorted(glob.glob(os.path.join(subject_dir, "*_run*.nii.gz")))

        if not run_files:
            run_files = sorted(glob.glob(os.path.join(subject_dir, "*.nii.gz")))

        if len(run_files) != n_runs:
            raise ValueError(
                f"{subject_id} has {len(run_files)} runs, expected {n_runs}"
            )

        subjects.append(SubjectFiles(subject_id=subject_id, run_files=run_files))

    return subjects


def preprocess_one_run_file(
    run_file: str,
    mask_path: str,
    trim: int,
    detrend: bool,
    standardize: bool,
    high_pass: float,
    t_r: float,
    smoothing_fwhm: float,
) -> np.ndarray:
    standardize_mode = "zscore_sample" if standardize else False

    masker = NiftiMasker(
        mask_img=mask_path,
        detrend=detrend,
        standardize=standardize_mode,
        high_pass=high_pass if high_pass > 0 else None,
        t_r=t_r,
        smoothing_fwhm=smoothing_fwhm if smoothing_fwhm > 0 else None,
    )

    data = masker.fit_transform(run_file)

    if data.ndim != 2:
        raise ValueError(f"NiftiMasker output must be 2D, got shape={data.shape}")

    if trim > 0:
        if data.shape[0] <= 2 * trim:
            raise ValueError(
                f"Timepoints too short after trim for {run_file}: "
                f"n_timepoints={data.shape[0]}, trim={trim}"
            )
        data = data[trim:-trim, :]

    return data.astype(np.float32, copy=False)


def load_language_run_matrix(
    subjects: Sequence[SubjectFiles],
    run_idx: int,
    mask_path: str,
    trim: int,
    detrend: bool,
    standardize: bool,
    high_pass: float,
    t_r: float,
    smoothing_fwhm: float,
    n_jobs: int,
) -> np.ndarray:
    run_files = [s.run_files[run_idx] for s in subjects]

    if n_jobs == 1:
        arrays = [
            preprocess_one_run_file(
                run_file=rf,
                mask_path=mask_path,
                trim=trim,
                detrend=detrend,
                standardize=standardize,
                high_pass=high_pass,
                t_r=t_r,
                smoothing_fwhm=smoothing_fwhm,
            )
            for rf in run_files
        ]
    else:
        arrays = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(preprocess_one_run_file)(
                run_file=rf,
                mask_path=mask_path,
                trim=trim,
                detrend=detrend,
                standardize=standardize,
                high_pass=high_pass,
                t_r=t_r,
                smoothing_fwhm=smoothing_fwhm,
            )
            for rf in run_files
        )

    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"Run {run_idx + 1} has inconsistent shapes across subjects: {shapes}"
        )

    return np.stack(arrays, axis=0)


def sample_candidate_groups(
    n_subjects: int,
    group_size: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if group_size > n_subjects:
        raise ValueError(f"group_size={group_size} > n_subjects={n_subjects}")

    if group_size == n_subjects:
        return np.arange(n_subjects, dtype=np.int32)[None, :]

    groups = np.empty((n_candidates, group_size), dtype=np.int32)

    for i in range(n_candidates):
        groups[i] = rng.choice(n_subjects, size=group_size, replace=False)

    seen = set()
    uniq = []

    for row in groups:
        key = tuple(sorted(row.tolist()))
        if key not in seen:
            seen.add(key)
            uniq.append(np.array(key, dtype=np.int32))

    return np.stack(uniq, axis=0)


def make_repeated_splits(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(seed)

    all_splits: List[List[Tuple[np.ndarray, np.ndarray]]] = []

    for group_idx in groups:
        group_size = group_idx.shape[0]

        if group_size % 2 != 0:
            raise ValueError(
                f"Split-half ISC requires an even group size, got {group_size}"
            )

        half = group_size // 2
        splits_for_group: List[Tuple[np.ndarray, np.ndarray]] = []

        for _ in range(n_splits):
            order = rng.permutation(group_size)
            idx_a = group_idx[order[:half]]
            idx_b = group_idx[order[half:]]
            splits_for_group.append((idx_a, idx_b))

        all_splits.append(splits_for_group)

    return all_splits


def voxelwise_corr_mean(x: np.ndarray, y: np.ndarray) -> float:
    """
    x, y: shape = (timepoints, voxels)

    Returns the mean Pearson correlation across voxels.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")

    x0 = x - x.mean(axis=0, keepdims=True)
    y0 = y - y.mean(axis=0, keepdims=True)

    numerator = np.sum(x0 * y0, axis=0, dtype=np.float64)

    x_norm = np.sum(x0 * x0, axis=0, dtype=np.float64)
    y_norm = np.sum(y0 * y0, axis=0, dtype=np.float64)
    denominator = np.sqrt(x_norm * y_norm)

    valid = denominator > 0

    if not np.any(valid):
        return 0.0

    corr = numerator[valid] / denominator[valid]

    return float(np.mean(corr))


def score_candidates_for_language(
    lang: str,
    subjects: Sequence[SubjectFiles],
    groups: np.ndarray,
    split_indices: List[List[Tuple[np.ndarray, np.ndarray]]],
    mask_path: str,
    n_runs: int,
    trim: int,
    detrend: bool,
    standardize: bool,
    high_pass: float,
    t_r: float,
    smoothing_fwhm: float,
    n_jobs: int,
) -> np.ndarray:
    """
    Score each candidate group using repeated split-half voxel-wise ISC.

    To reduce memory use, this function loads one run at a time.
    """
    n_groups = groups.shape[0]
    n_splits = len(split_indices[0])

    score_sums = np.zeros(n_groups, dtype=np.float64)

    for run_idx in range(n_runs):
        t_run = time.time()

        print(
            f"[INFO] {lang.upper()} loading run {run_idx + 1}/{n_runs}...",
            flush=True,
        )

        run_mat = load_language_run_matrix(
            subjects=subjects,
            run_idx=run_idx,
            mask_path=mask_path,
            trim=trim,
            detrend=detrend,
            standardize=standardize,
            high_pass=high_pass,
            t_r=t_r,
            smoothing_fwhm=smoothing_fwhm,
            n_jobs=n_jobs,
        )

        print(
            f"[INFO] {lang.upper()} run {run_idx + 1}: "
            f"matrix shape={run_mat.shape} loaded in {time.time() - t_run:.1f}s",
            flush=True,
        )

        for group_i in range(n_groups):
            split_scores = []

            for idx_a, idx_b in split_indices[group_i]:
                half_a = run_mat[idx_a].mean(axis=0)
                half_b = run_mat[idx_b].mean(axis=0)

                split_scores.append(voxelwise_corr_mean(half_a, half_b))

            score_sums[group_i] += float(np.mean(split_scores))

        del run_mat

    scores = score_sums / n_runs

    return scores


def find_best_triplet(
    en_scores: np.ndarray,
    cn_scores: np.ndarray,
    fr_scores: np.ndarray,
) -> Tuple[int, int, int, float]:
    cn_order = np.argsort(cn_scores)
    cn_sorted = cn_scores[cn_order]

    best = (0, 0, 0, float("inf"))

    for i_fr, s_fr in enumerate(fr_scores):
        for i_en, s_en in enumerate(en_scores):
            lo = min(s_en, s_fr)
            hi = max(s_en, s_fr)
            base = hi - lo

            pos = bisect.bisect_left(cn_sorted, lo)

            candidate_positions = []

            if pos < len(cn_sorted):
                candidate_positions.append(pos)

            if pos > 0:
                candidate_positions.append(pos - 1)

            if pos < len(cn_sorted) and cn_sorted[pos] <= hi:
                gap = base
                i_cn = int(cn_order[pos])

                if gap < best[3]:
                    best = (i_en, i_cn, i_fr, float(gap))

                continue

            for p in candidate_positions:
                s_cn = cn_sorted[p]
                gap = max(s_en, s_cn, s_fr) - min(s_en, s_cn, s_fr)

                if gap < best[3]:
                    i_cn = int(cn_order[p])
                    best = (i_en, i_cn, i_fr, float(gap))

    return best


def group_ids(subject_ids: Sequence[str], group_idx: np.ndarray) -> List[str]:
    return [subject_ids[i] for i in sorted(group_idx.tolist())]


def save_ids(path: str, ids: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sid in ids:
            f.write(f"{sid}\n")


def save_candidate_scores(
    path: str,
    subject_ids: Sequence[str],
    groups: np.ndarray,
    scores: np.ndarray,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("candidate_idx\tisc_score\tparticipant_ids\n")

        for i, (group_idx, score) in enumerate(zip(groups, scores)):
            ids = group_ids(subject_ids, group_idx)
            f.write(f"{i}\t{float(score):.10f}\t{','.join(ids)}\n")


def main() -> None:
    args = parse_args()
    make_dir(args.out_dir)

    t0 = time.time()

    common_mask_path, common_mask_voxels = validate_mask(args.common_mask)

    print(
        f"[INFO] Using existing common mask: {common_mask_path} "
        f"(voxels={common_mask_voxels})",
        flush=True,
    )

    subject_files: Dict[str, List[SubjectFiles]] = {}
    subject_ids: Dict[str, List[str]] = {}

    for lang in LANGS:
        subject_files[lang] = collect_subject_files(
            home_folder=args.home_folder,
            lang=lang,
            n_runs=args.n_runs,
        )
        subject_ids[lang] = [s.subject_id for s in subject_files[lang]]

        print(
            f"[INFO] Found {len(subject_files[lang])} {lang.upper()} subjects",
            flush=True,
        )

    rng = np.random.default_rng(args.seed)

    groups: Dict[str, np.ndarray] = {}
    split_indices: Dict[str, List[List[Tuple[np.ndarray, np.ndarray]]]] = {}
    scores: Dict[str, np.ndarray] = {}

    for lang_i, lang in enumerate(LANGS):
        n_subjects = len(subject_files[lang])

        groups[lang] = sample_candidate_groups(
            n_subjects=n_subjects,
            group_size=args.group_size,
            n_candidates=args.n_candidates,
            rng=rng,
        )

        split_indices[lang] = make_repeated_splits(
            groups=groups[lang],
            n_splits=args.n_splits,
            seed=args.seed + 1000 * (lang_i + 1),
        )

        print(
            f"[INFO] {lang.upper()} candidates={groups[lang].shape[0]} "
            f"(requested={args.n_candidates}); repeated splits={args.n_splits}",
            flush=True,
        )

    for lang_i, lang in enumerate(LANGS):
        t_lang = time.time()

        print(
            f"[INFO] Scoring {lang.upper()} candidates with voxel-wise ISC...",
            flush=True,
        )

        scores[lang] = score_candidates_for_language(
            lang=lang,
            subjects=subject_files[lang],
            groups=groups[lang],
            split_indices=split_indices[lang],
            mask_path=common_mask_path,
            n_runs=args.n_runs,
            trim=args.trim,
            detrend=args.detrend,
            standardize=args.standardize,
            high_pass=args.high_pass,
            t_r=args.t_r,
            smoothing_fwhm=args.smoothing_fwhm,
            n_jobs=args.n_jobs,
        )

        save_candidate_scores(
            path=os.path.join(args.out_dir, f"candidate_scores_{lang}.tsv"),
            subject_ids=subject_ids[lang],
            groups=groups[lang],
            scores=scores[lang],
        )

        print(
            f"[INFO] Scored {lang.upper()} in {time.time() - t_lang:.1f}s",
            flush=True,
        )

    i_en, i_cn, i_fr, best_gap = find_best_triplet(
        en_scores=scores["en"],
        cn_scores=scores["cn"],
        fr_scores=scores["fr"],
    )

    matched_idx = {
        "en": groups["en"][i_en],
        "cn": groups["cn"][i_cn],
        "fr": groups["fr"][i_fr],
    }

    matched_ids = {
        lang: group_ids(subject_ids[lang], matched_idx[lang])
        for lang in LANGS
    }

    matched_scores = {
        "en": float(scores["en"][i_en]),
        "cn": float(scores["cn"][i_cn]),
        "fr": float(scores["fr"][i_fr]),
    }

    summary = {
        "method": "voxelwise_repeated_split_half_isc_matching",
        "common_mask_path": common_mask_path,
        "common_mask_voxels": common_mask_voxels,
        "params": {
            "home_folder": args.home_folder,
            "group_size": args.group_size,
            "n_candidates_requested": args.n_candidates,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "n_runs": args.n_runs,
            "trim": args.trim,
            "high_pass": args.high_pass,
            "t_r": args.t_r,
            "smoothing_fwhm": args.smoothing_fwhm,
            "detrend": args.detrend,
            "standardize": args.standardize,
            "n_jobs": args.n_jobs,
        },
        "language_subject_counts": {
            lang: len(subject_ids[lang]) for lang in LANGS
        },
        "candidate_counts_actual": {
            lang: int(groups[lang].shape[0]) for lang in LANGS
        },
        "matched_scores": matched_scores,
        "matched_gap_max_minus_min": float(best_gap),
        "matched_participant_ids": matched_ids,
    }

    summary_path = os.path.join(args.out_dir, "matched_group_isc_summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_ids(
        os.path.join(args.out_dir, "matched_average_subject_ids_en.txt"),
        matched_ids["en"],
    )
    save_ids(
        os.path.join(args.out_dir, "matched_average_subject_ids_cn.txt"),
        matched_ids["cn"],
    )
    save_ids(
        os.path.join(args.out_dir, "matched_average_subject_ids_fr.txt"),
        matched_ids["fr"],
    )

    print(f"[INFO] Done in {time.time() - t0:.1f}s", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
