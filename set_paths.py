import numpy as np
import os
from pathlib import Path

# list of folders used in the study
# repository root, containing code and generated metadata
home_folder = str(Path(__file__).resolve().parent)
# output root for generated average-subject files
output_root = os.path.join(home_folder, 'lpp_averaged_subject')
# output root for masks and matched participant IDs
masks_root = os.path.join(home_folder, 'masks')
matched_group_isc_dir = os.path.join(home_folder, 'matched_group_isc')
# path to Le Petit Prince fMRI corpus, downloaded from https://doi.org/10.18112/openneuro.ds003643.v2.0.7
lpp_path = os.environ.get('LPP_PATH', '/data/datasets/lpp-fmri/ds003643')

# fmri data
fmri_data = os.path.join(lpp_path, 'derivatives')
# annotations, used for aligning text and speech
annotation_folder = os.path.join(lpp_path, 'annotation')

# # location of the GloVe embeddings
# glove_embeddings_path = os.path.join(home_folder, 'glove.6B.300d.txt')
# # location of activations from the various llms
# llms_activations = os.path.join(home_folder, 'llms_activations')
# # location of brain correlations for each model, for each layer
# llms_brain_correlations = os.path.join(home_folder, 'llms_brain_correlations')
# llms_brain_correlations_individual = os.path.join(home_folder, 'llms_brain_correlations_individual')
# nii files for the roi masks
# roi_masks = os.path.join(home_folder, 'roi_masks')

# all figures in the paper
figures_folder = os.path.join(home_folder, 'figures')
#location of the average subject fmri data
lpp_average = os.path.join(home_folder, 'lpp_average')

# common mask used for matched average-subject generation
common_mask_path = os.path.join(masks_root, 'mask_lpp_all.nii.gz')

n_runs = 9
t_r = 2 #s



# helpers
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def standardize(v, axis=0):
    return (v - np.mean(v, axis=axis, keepdims=True)) / np.std(v, axis=axis, keepdims=True)
