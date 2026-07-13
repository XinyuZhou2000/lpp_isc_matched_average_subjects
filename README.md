# lpp isc-matched average subjects

The dataset ["Le Petit Prince"](https://openneuro.org/datasets/ds003643/versions/2.0.7) (Li, Hale & Palier, 2025) 
provides 3T functional magnetic resonance imaging (fMRI) data from 49 English, 35 Chinese and 28 French participants who listened an audiobook of *Le Petit Prince*, spliced into 9 segments of ~10min.  

Because the participants in a given language listened exactly to the same stimuli, instead of running analyses on individual data, it is sometimes relevant to run them on an average subject obtained by spatially aligning and averaging all the individual functional time series, for each of the 9 segments. Thus, the average subjects for the LPP project computed over the full sample in all three languages (English, Chinese, French) are available at <https://github.com/l-bg/llms_brain_lateralization>.

However, given that the sample sizes in the LPP dataset differ across languages, these average subjects are **not** appropriate for between language comparisons. 

Indeed, when average subjects are built from all available participants, the English average subject is based on substantially more participants than the French average subject. This difference could confound later comparisons of encoding performance across languages. Even if the number of subjects is matched, the quality of the measurements and the participants' engagement in the task can lead to different inter-subject correlations (ISC). To address this issue, we here provide three averaged subjects (one for each language) matched in terms of of number of participants (n=28) and mean ISC. More precisely, this repository contains:

- [mask_lpp_all.nii.gz](./masks/mask_lpp_all.nii.gz): a binary mask common to all three languages.
- three folders, one for each language, containing 9 nifti files obtained by averaging the 28 participants listed in the accompanying `summary.json` file, and the preprocessing settings used to generate the files:
  - [English average subject](./lpp_averaged_subject/lpp_en_average_subject_matched/)
  - [Chinese average subject](./lpp_averaged_subject/lpp_cn_average_subject_matched/)
  - [French average subject](./lpp_averaged_subject/lpp_fr_average_subject_matched/)
- Three inter-subject correlation maps:
   -  [English group ISC map](./isc_maps/isc_en_matched_n28_seed1234.nii.gz)
   -  [French group ISC map](./isc_maps/isc_cn_matched_n28_seed1234.nii.gz)
   -  [Chines group ISC map](./isc_maps/isc_fr_matched_n28_seed1234.nii.gz)
- Python code used to select the subjects, compute the ISC and average subjects.
- An example downstream use case is described in the M2 thesis of Xinyu Zhou, [M2_Dissertation_XinyuZhou.pdf](./papers/M2_Dissertation_XinyuZhou.pdf). The thesis uses these ISC-matched average subjects for cross-lingual transfer analyses, so that English, Chinese, and French fMRI targets can be compared while controlling for participant count and group-level reliability.

## Procedure

Our procedure (described in details in [paper](./papers/isc_matched_average_subjects.pdf) to compute these averaged subjects was as follows:

0. download lpp dataset from <https://openneuro.org/datasets/ds003643/versions/2.0.7>
1. resample all spatially normalized bold volumes to 4x4x4mm using [resample_fmri_data.py](resample_fmri_data.py)
2. compute a common mask using [compute_common_mask](compute_common_mask.py). This script is modified from [`compute_mask.py`](https://github.com/l-bg/llms_brain_lateralization/blob/main/compute_mask.py) in `llms_brain_lateralization` and keeps the original language-specific mask option.
3. for each of the 9 runs, the bold data was detrended, standardize, trimmed (deleting the first 10 and the last 10 scans), highpass filter at 1/128Hz, then averaged accross the 28 French individual to produce a single [French average subject](./lpp_averaged_subject/lpp_fr_average_subject_matched/).
4. select 28 English and 28 Chinese participants so that the mean ISC of these groups are the same as for the French group, and create the single [English average subject](./lpp_averaged_subject/lpp_en_average_subject_matched/) and [Chinese average subject](./lpp_averaged_subject/lpp_cn_average_subject_matched/).

## How to reproduce all the ISC-matched average subjects

0. Install the Python dependencies first:

   ```bash
   pip install -r requirements.txt
   ```

1. Download the original LPP dataset from OpenNeuro, then edit `lpp_path` in
   `set_paths.py` or set the `LPP_PATH` environment variable.

2. Resample spatially normalized BOLD images to 4 x 4 x 4 mm:

   ```bash
   python resample_fmri_data.py --lang en
   python resample_fmri_data.py --lang cn
   python resample_fmri_data.py --lang fr
   ```

3. Compute one common EN/CN/FR mask from all resampled images:

   ```bash
   python compute_common_mask.py --lang all
   ```

   This writes the downstream mask to `masks/mask_lpp_all.nii.gz`.

4. Match 28-participant groups by group-level ISC:

   ```bash
   python match_group_isc.py
   ```

   The matched participant groups are selected with a fixed random seed
   (`--seed 1234` by default), which is recorded in the matching output
   summary.

   For each of the 9 runs, the BOLD data are detrended, standardized,
   high-pass filtered at 1/128 Hz, smoothed with an 8 mm FWHM kernel, and
   trimmed by deleting the first 10 and last 10 scans. Candidate 28-subject
   groups are randomly sampled for each language. Each candidate receives a
   group-level ISC score computed from 10 repeated split-half estimates:
   subjects are averaged within each half, Pearson correlation is computed
   separately for each voxel, and voxel-wise correlations are averaged across
   voxels, splits, and runs.

5. Build the final average subjects from the matched participant IDs:

   ```bash
   python compute_average_subject_from_matched_ids.py
   ```

   This final reconstruction step does not trim the time series by default
   (`--trim 0`). Trimming is used only during the ISC-matching/group-selection
   step above.

6. Compute the three final voxel-wise ISC maps for the matched groups:

   ```bash
   python compute_final_isc_maps.py
   ```

   The final mean ISC values inside the common mask are:

   | Language | Mean ISC |
   | --- | ---: |
   | English | 0.3019 |
   | Chinese | 0.3027 |
   | French | 0.3005 |


## Matched Participant Groups

The final matched groups contain 28 participants for each language. The same
IDs are stored in `matched_group_isc/matched_average_subject_ids_*.txt`. These
groups were selected with fixed seed `1234`.

### C.1 English

The selected English participants were:

```text
sub-EN061 sub-EN062 sub-EN064 sub-EN067
sub-EN069 sub-EN070 sub-EN073 sub-EN074
sub-EN075 sub-EN076 sub-EN082 sub-EN083
sub-EN084 sub-EN087 sub-EN089 sub-EN093
sub-EN095 sub-EN097 sub-EN098 sub-EN099
sub-EN100 sub-EN101 sub-EN105 sub-EN106
sub-EN110 sub-EN113 sub-EN114 sub-EN115
```

### C.2 Chinese

The selected Chinese participants were:

```text
sub-CN001 sub-CN002 sub-CN003 sub-CN005
sub-CN006 sub-CN009 sub-CN010 sub-CN011
sub-CN013 sub-CN015 sub-CN016 sub-CN019
sub-CN020 sub-CN021 sub-CN022 sub-CN023
sub-CN024 sub-CN025 sub-CN027 sub-CN028
sub-CN029 sub-CN030 sub-CN031 sub-CN032
sub-CN033 sub-CN034 sub-CN036 sub-CN037
```

### C.3 French

The selected French participants were:

```text
sub-FR001 sub-FR002 sub-FR003 sub-FR004
sub-FR005 sub-FR006 sub-FR007 sub-FR008
sub-FR009 sub-FR010 sub-FR011 sub-FR012
sub-FR013 sub-FR014 sub-FR015 sub-FR016
sub-FR017 sub-FR018 sub-FR019 sub-FR020
sub-FR022 sub-FR023 sub-FR024 sub-FR025
sub-FR026 sub-FR028 sub-FR029 sub-FR030
```

## References

> Zhou, X., Pallier, C., & Bonnasse-Gahot, L. [ISC-matched average subjects for the multilingual Le Petit Prince fMRI dataset](./papers/isc_matched_average_subjects.pdf).

> Zhou, X. [Dissertation](./papers/M2_Dissertation_XinyuZhou.pdf).

> Bonnasse-Gahot, L., & Pallier, C. (2024). [fMRI predictors based on language models of increasing complexity recover brain left lateralization](https://proceedings.neurips.cc/paper_files/paper/2024/file/e28e19d00b23fe0265f433fa05a96b06-Paper-Conference.pdf). In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, & C. Zhang (Eds.), *Advances in Neural Information Processing Systems* (Vol. 37, pp. 125231-125263). Curran Associates, Inc.

> Bonnasse-Gahot, L., & Pallier, C. (2024). Source code for "fMRI Predictors Based on Language Models of Increasing Complexity Recover Brain Left Lateralization." (Version 1.0.0) [Computer software]. <https://doi.org/10.5281/zenodo.19097232>

> Jixing Li, John Hale, and Christophe Pallier (2025). Le Petit Prince: A multilingual fMRI corpus using ecological stimuli. OpenNeuro. [Dataset] <https://doi.org/10.18112/openneuro.ds003643.v2.0.7>

> Li, J., Bhattasali, S., Zhang, S., Franzluebbers, B., Luh, W.-M., Spreng, R. N., Brennan, J. R., Yang, Y., Pallier, C., & Hale, J. (2022). [Le Petit Prince multilingual naturalistic fMRI corpus](https://doi.org/10.1038/s41597-022-01625-7). *Scientific Data*, 9(1), Article 1.
