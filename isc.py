#!/usr/bin/env python3
"""
Group-wise ISC Calculation
-----------------------------------------
Modes:
1. Random (--mode random):
   - For each n, randomly sample 2n subjects from the entire pool for each iteration.
   - Calculates run-averaged voxel-wise ISC.

2. Top/Bottom (--mode topbottom):
   - For each n, select the fixed Top 2n and Bottom 2n subjects based on quiz scores.
   - Iterations perform random split-half shuffles on these fixed groups.

Usage:
   python isc.py --mode random --lang EN --n_iter 30
   python isc.py --mode topbottom --lang EN --n_iter 30
"""

import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory
from nilearn.maskers import NiftiMasker

import set_paths as spath

# ---------------- Paths & Constants ---------------- #
DATA_ROOT       = spath.home_folder
MASK_EN         = os.path.join(spath.masks_root, 'mask_lpp_en.nii.gz')
MASK_CN         = os.path.join(spath.masks_root, 'mask_lpp_cn.nii.gz')
MASK_FR         = os.path.join(spath.masks_root, 'mask_lpp_fr.nii.gz')
PARTICIPANT_TSV = os.path.join(spath.lpp_path, 'participants.tsv')
OUTPUT_DIR      = os.path.join(spath.home_folder, 'isc_maps')
CACHE_DIR       = os.path.join(spath.home_folder, '.cache', 'isc')

LANG_DIRS = {
    "EN": "lpp_en_resampled",
    "CN": "lpp_cn_resampled",
    "FR": "lpp_fr_resampled"
}
LANG_MASKS = {
    "EN": MASK_EN,
    "CN": MASK_CN,
    "FR": MASK_FR,
}

RUNS   = 9
TR     = 2.0
N_LIST = [4, 8, 12, 16, 20, 24]

os.makedirs(OUTPUT_DIR, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)

# ---------------- Helper Functions ---------------- #

def get_available_subjects(lang):
    """Scan the directory to find available subject IDs."""
    p = re.compile(rf'sub-{lang}(\d{{3}})')
    lang_dir = os.path.join(DATA_ROOT, LANG_DIRS[lang])
    if not os.path.exists(lang_dir):
        raise FileNotFoundError(f"Directory not found: {lang_dir}")
    return sorted(int(m[1]) for s in os.listdir(lang_dir) if (m := p.match(s)))

def get_data_path(lang, sub, run):
    """Construct file path for a specific subject run."""
    label = f'sub-{lang}{sub:03d}'
    return os.path.join(DATA_ROOT, LANG_DIRS[lang],
                        label, f'{label}_run{run+1}.nii.gz')

@memory.cache(ignore=['masker'])
def load_and_preproc(lang, sub, run, masker, mask_tag):
    """Load Nifti file and apply masker (cached)."""
    fp = get_data_path(lang, sub, run)
    if not os.path.exists(fp):
        return None
    # Remove first and last 10 TRs
    return masker.transform(fp)[10:-10]

def compute_run_average(lang, subs, run, masker, mask_tag):
    """Load data for a group of subjects and average them for a specific run."""
    stack = [load_and_preproc(lang, s, run, masker, mask_tag) for s in subs]
    stack = [x for x in stack if x is not None]
    if not stack:
        raise ValueError(f"Missing data for run {run+1}")
    return np.mean(stack, axis=0)

def voxelwise_corr(a, b):
    """Compute column-wise (voxel-wise) correlation between two 2D arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    numer = np.sum(a * b, axis=0)
    return np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=np.float64),
        where=denom > 0,
    ).astype(np.float32)

def groupwise_isc_once(lang, subs, masker, mask_tag, rng):
    """
    Core calculation:
    1. Shuffle subjects
    2. Split into two halves
    3. Compute average timecourse for each half per run
    4. Correlate and average across runs
    """
    # Ensure we work on a copy to avoid side effects if list is reused
    subs = list(subs)
    rng.shuffle(subs)
    mid = len(subs) // 2
    
    corrs = []
    for r in range(RUNS):
        d1 = compute_run_average(lang, subs[:mid], r, masker, mask_tag)
        d2 = compute_run_average(lang, subs[mid:], r, masker, mask_tag)
        corrs.append(voxelwise_corr(d1, d2))
    
    return {'ids': subs, 'isc': np.mean(corrs, axis=0)}

def build_ranked_lists(lang, seed):
    """Parse TSV, handle ties, and return sorted subject IDs."""
    df = pd.read_csv(PARTICIPANT_TSV, sep='\t')
    df = df[df['participant_id'].str.contains(lang)]
    df['score'] = pd.to_numeric(df['correct_quiz_questions'], errors='coerce')
    df = df.dropna(subset=['score'])
    df['sid'] = df['participant_id'].str.extract(r'(\d{3})').astype(int)
    
    # Random tie-breaking
    rng = np.random.default_rng(seed)
    df['tie'] = rng.random(len(df))
    
    top = (df.sort_values(['score', 'tie'], ascending=[False, True])['sid'].tolist())
    bottom = (df.sort_values(['score', 'tie'], ascending=[True, True])['sid'].tolist())
    return top, bottom

# ---------------- Main Logic ---------------- #

def main():
    global DATA_ROOT, PARTICIPANT_TSV, OUTPUT_DIR, LANG_MASKS

    parser = argparse.ArgumentParser(description="Unified Group-wise ISC")
    parser.add_argument('--mode', type=str, required=True, choices=['random', 'topbottom'],
                        help="Execution mode: 'random' sampling or 'topbottom' ranking")
    parser.add_argument('--seed', type=int, default=1, help='Global RNG seed')
    parser.add_argument('--n_iter', type=int, default=30, help='Iterations per n')
    parser.add_argument('--lang', type=str, default='EN', choices=['EN', 'CN', 'FR'],
                        help='Language (EN/CN/FR)')
    parser.add_argument('--data-root', type=str, default=DATA_ROOT,
                        help='Root containing lpp_<lang>_resampled folders')
    parser.add_argument('--participant-tsv', type=str, default=PARTICIPANT_TSV,
                        help='participants.tsv from the original LPP dataset')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Directory where ISC result pickle files are saved')
    parser.add_argument('--mask-dir', type=str, default=spath.masks_root,
                        help='Directory containing mask_lpp_en/cn/fr.nii.gz')
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    PARTICIPANT_TSV = args.participant_tsv
    OUTPUT_DIR = args.output_dir
    LANG_MASKS = {
        "EN": os.path.join(args.mask_dir, 'mask_lpp_en.nii.gz'),
        "CN": os.path.join(args.mask_dir, 'mask_lpp_cn.nii.gz'),
        "FR": os.path.join(args.mask_dir, 'mask_lpp_fr.nii.gz'),
    }

    if args.mode == 'topbottom' and args.lang == 'FR':
        raise ValueError("Top/bottom ISC requires comprehension quiz scores and is only supported for EN/CN.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Starting: Mode={args.mode}, Lang={args.lang}, Seed={args.seed} ---")

    # Initialize language-specific masker
    mask_img = LANG_MASKS[args.lang]
    if not os.path.exists(mask_img):
        raise FileNotFoundError(f"Mask not found for {args.lang}: {mask_img}")
    # cache key marker to prevent stale cache reuse when mask changes
    mask_tag = os.path.basename(mask_img)
    masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=8,
                         detrend=True, standardize=True,
                         low_pass=0.2, high_pass=0.01,
                         t_r=TR, memory='nilearn_cache',
                         memory_level=1, verbose=0).fit()

    # Get available subjects
    avail_subjects = set(get_available_subjects(args.lang))
    avail_list = sorted(list(avail_subjects))
    
    if len(avail_list) < 2 * max(N_LIST):
        print(f"Warning: Only {len(avail_list)} subjects available. "
              f"Max N={max(N_LIST)} requires {2*max(N_LIST)} subjects.")

    results = {}

    # ---------------- MODE: RANDOM ---------------- #
    if args.mode == 'random':
        for n in N_LIST:
            need = 2 * n
            if len(avail_list) < need:
                print(f"[Skip] n={n} (need {need}, have {len(avail_list)})")
                continue

            print(f"n={n}: random sampling x {args.n_iter}")

            def one_iter_random(i):
                # Critical: Sample NEW subjects for every iteration
                rng = np.random.default_rng(args.seed + i)
                pool = rng.choice(avail_list, size=need, replace=False).tolist()
                return groupwise_isc_once(args.lang, pool, masker, mask_tag, rng)

            results[n] = Parallel(n_jobs=-1)(
                delayed(one_iter_random)(i) for i in range(args.n_iter)
            )

    # ---------------- MODE: TOP/BOTTOM ---------------- #
    elif args.mode == 'topbottom':
        # Get ranked lists
        top_all, bottom_all = build_ranked_lists(args.lang, args.seed)
        top_all = [s for s in top_all if s in avail_subjects]
        bottom_all = [s for s in bottom_all if s in avail_subjects]
        
        results = {'top': {}, 'bottom': {}}

        for n in N_LIST:
            need = 2 * n
            if len(top_all) < need or len(bottom_all) < need:
                print(f"[Skip] n={n} (need {need}, have Top:{len(top_all)}/Bot:{len(bottom_all)})")
                continue
            
            # Slice fixed pools
            top_pool = top_all[:need]
            bottom_pool = bottom_all[:need]

            print(f"n={n}: computing TOP & BOTTOM ...")

            def one_iter_fixed(i, pool):
                # Pool is fixed, just reshuffling inside groupwise_isc_once
                rng = np.random.default_rng(args.seed + 100000 + i)
                return groupwise_isc_once(args.lang, pool, masker, mask_tag, rng)

            # Parallel execution for Top
            results['top'][n] = Parallel(n_jobs=-1)(
                delayed(one_iter_fixed)(i, top_pool) for i in range(args.n_iter)
            )
            # Parallel execution for Bottom
            results['bottom'][n] = Parallel(n_jobs=-1)(
                delayed(one_iter_fixed)(i, bottom_pool) for i in range(args.n_iter)
            )

    # ---------------- Save Results ---------------- #
    out_name = f"isc_{args.lang.lower()}_{args.mode}_n{','.join(map(str, N_LIST))}_" \
               f"{args.n_iter}iter_seed{args.seed}.pkl"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Done. Saved to -> {out_path}")

if __name__ == "__main__":
    main()
