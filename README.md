# ST-GEARS
[![DOI](https://zenodo.org/badge/714884140.svg)](https://zenodo.org/doi/10.5281/zenodo.13131712)
[![stars](https://img.shields.io/github/stars/STOmics/ST-GEARS)](https://img.shields.io/github/stars/STOmics/ST-GEARS) 

A strong 3D reconstruction tool for Spatial Transcriptomics, with accurate position alignment plus distortion correction!
![fig1](https://github.com/STOmics/ST-GEARS/assets/96898334/6617eaaf-d6f5-4966-b7da-631d8c08e79d)

ST-GEARS consists of methods to compute anchors, to rigidly align and to elastically registrate sections. Specifically, 

`serial_align` computes mappings between adjacent sections in serial, using Fused-Gromov Wasserstein Optimal Transport with our innovatie Distributive Constraints

`stack_slices_pairwise_rigid` rigidly aligns sections using Procrustes Analysis

`stack_slices_pairwise_elas_field` eliminates distorsions through Gaussian Smoothed Elastic Fields. Validity is proved mathematically

### Article
Our article is published at: [Nature Communications](https://www.nature.com/articles/s41467-024-51935-0)

### News
We just added granularity adjusting to README. It really helps when you have over 3000 in any one of your sections.

### Installation
```python
git clone https://github.com/STOmics/ST-GEARS.git

cd ST-GEARS

pip install .

pip install -r requirements.txt
```
Installation was tested on both Windows and Linux os. Typical installation time is less than 3 mins.

### Input and Output
To use ST-GEARS to recover geospatial profile in 3D, you need gene expression matrix, spatial information and a rough grouping of observations. 

Structure above input in a list of anndata.Anndata of below. Also find rigid and elastic registration results in this structure:
![image](https://github.com/user-attachments/assets/861a99d7-004b-42d4-9689-6289c0328a9a)


### Our Recipe
```python
import os
import st_gears
import warnings

warnings.filterwarnings('ignore')

grp_col = 'annotation'  # name of column in adata.obs that stores grouping information, either cluster or annotation information
binning_on = True  # only turn granularity adjusting on, when you have over 3000 spots in any one of the sections.

# set uniform_weight to False and filter_by_label to True if you trust in your groupping result, and sections are not too far from each other. Otherwise set uniform_weight to True and filter_by_label to False.
uniform_weight = False  # False if using Distributive Constraints, else True
filter_by_label = True  # Filter groups of spot that do not co-occur in two sections when computing anchors

# binning
if binning_on:
    step = 2  # step size for binning. Check your xy coordinates first. For example, when adjacent spots roughly sits 10 away from each other, setting step to 20 would decrease spots for 4 times in binning result. Remain roughly at least 200 spots in each section after binning.
    slice_srk_li = [st_gears.binning(slice_, grp_col, step) for slice_ in slicesl]  # 'slice_srk_li' means 'shrinked slice list'
else:
    slice_srk_li = slicesl

# Compute anchors
anncell_cid = st_gears.helper.gen_anncell_cid_from_all(slice_srk_li, grp_col)
pili, tyscoreli, alphali, regis_ilist, ali, bli = st_gears.serial_align(slice_srk_li, anncell_cid, label_col=grp_col,
                                                                        start_i=0, end_i=len(slicesl)-1,  # index of start and end section from slicesl to be aligned
                                                                        tune_alpha_li=[0.8, 0.2, 0.05, 0.013],  # regularization factor list, recommend to fill values exponentially change among 0 and 1. Longer list indicates finer tuning resolution
                                                                        numItermax=200,  # max number of iteration during optimization
                                                                        dissimilarity_val='kl', dissimilarity_weight_val='kl',
                                                                        uniform_weight=uniform_weight, map_method_dis2wei='logistic',
                                                                        filter_by_label=filter_by_label,
                                                                        use_gpu=False, verbose=True)  # show each iteration or not


# Rigid registration
fil_pc = 20  # not recommended to change # fil_pc_rigid / 100 * (maximum_probability - minimum_probability) + minimum_probability is set as theshhold to filter anchors
slice_srk_li = st_gears.stack_slices_pairwise_rigid([slice_srk_li[i] for i in regis_ilist],
						     pili,
						     label_col=grp_col,
                                                     fil_pc=fil_pc,
						     filter_by_label=filter_by_label)


# elastic registration
if binning_on:
    pixel_size = step  # pixel step size of elastic field. 
else:
    pixel_size = 1  # if not binned, input rough distance between adjacent spots on each section. 
sigma = 1  # not recommended to change. kernel size of Gaussian Filters, with a higher value indicating a smoother elastic field
slice_srk_li = st_gears.stack_slices_pairwise_elas_field([slice_srk_li[i] for i in regis_ilist],
                                                          pili,
                                                          label_col=grp_col,
                                                          pixel_size=pixel_size,
						          fil_pc=fil_pc,
                                                          filter_by_label=filter_by_label,
						          sigma=sigma)

# interpolate reconstruction result, back to original resolution
if binning_on:
    slicesl = st_gears.interpolate(slice_srk_li, slicesl)
else:
    slicesl = slice_srk_li
```

### Demo Tutorial
https://github.com/STOmics/ST-GEARS/blob/main/demo.ipynb
Expected run time on a typical quad core Windows in 5 mins.
