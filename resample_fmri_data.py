import numpy as np
import glob
import os
import argparse
from tqdm.auto import tqdm
from nilearn.image import resample_img

import set_paths as spath
from set_paths import make_dir


parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='en',
                    help='language: en, fr or cn')
args = parser.parse_args()
lang = args.lang.lower()

assert lang in ['en', 'fr', 'cn'], 'This language is not available. Please choose between en, fr or cn.'

target_affine = np.array([[   4.,    0.,    0.,  -72.],
                          [   0.,    4.,    0., -106.],
                          [   0.,    0.,    4.,  -64.],
                          [   0.,    0.,    0.,    1.]])

target_shape = (37, 46, 38)

output_path = os.path.join(spath.home_folder, 'lpp_{}_resampled'.format(lang))

subject_list = np.sort(glob.glob(os.path.join(spath.fmri_data, 'sub-{}*'.format(lang.upper()))))

for sub_id in tqdm(subject_list):
    sub_name = os.path.basename(sub_id)
    make_dir(os.path.join(output_path, sub_name))
    fmri_imgs_sub = sorted(glob.glob(os.path.join(sub_id, 'func/*.nii.gz')))
    for run, fmri_imgs_sub_run in enumerate(fmri_imgs_sub):
        img_resampled = resample_img(fmri_imgs_sub_run, 
                                     target_affine=target_affine, 
                                     target_shape=target_shape)
        img_resampled.to_filename(os.path.join(output_path,
                                               sub_name,
                                               '{}_run{}.nii.gz'.format(sub_name, run+1)))
