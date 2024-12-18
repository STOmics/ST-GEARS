# ST-GEARS
[![DOI](https://zenodo.org/badge/714884140.svg)](https://zenodo.org/doi/10.5281/zenodo.13131712)
[![Github All Releases](https://img.shields.io/github/downloads/<STOmics>/<ST-GEARS>/total.svg)]()

A strong 3D reconstruction tool for Spatial Transcriptomics, with accurate position alignment plus distortion correction!
![fig1](https://github.com/STOmics/ST-GEARS/assets/96898334/6617eaaf-d6f5-4966-b7da-631d8c08e79d)

ST-GEARS consists of methods to compute anchors, to rigidly align and to elastically registrate sections. Specifically, 

`serial_align` computes mappings between adjacent sections in serial, using Fused-Gromov Wasserstein Optimal Transport with our innovatie Distributive Constraints

`stack_slices_pairwise_rigid` rigidly aligns sections using Procrustes Analysis

`stack_slices_pairwise_elas_field` eliminates distorsions through Gaussian Smoothed Elastic Fields. Validity is proved mathematically

### Article
Read our article at: [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.12.09.570320v1)!

### News
we fixed bug of package importing, hence running it should be no problem now.

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
![dataformat](https://github.com/STOmics/ST-GEARS/assets/96898334/ffa6dfbd-1b25-4399-82d3-fc64b012fda3)

### Quick Start
```python
import os
import st_gears
import warnings

warnings.filterwarnings('ignore')

# Compute anchors
uniform_weight = False  # Set to True if using uniform weight, set to False if using Distributive Constraints
filter_by_label = True  # Filter groups of spot that do not co-occur in two sections when computing anchors
grp_col = 'annotation'  # name of column in adata that stores grouping information
anncell_cid = st_gears.helper.gen_anncell_cid_from_all(slicesl, grp_col)
pili, tyscoreli, alphali, regis_ilist, ali, bli = st_gears.serial_align(slicesl, anncell_cid, label_col=grp_col,
                                                                        start_i=0, end_i=len(slicesl)-1,  # index of start and end section from slicesl to be aligned
                                                                        tune_alpha_li=[0.8, 0.2, 0.05, 0.013],  # regularization factor list, recommend to fill values exponentially change among 0 and 1. Higher number of elements indicates finer tuning resolution
                                                                        numItermax=150,  # max number of iteration during optimization
                                                                        uniform_weight=uniform_weight,
                                                                        filter_by_label=filter_by_label,
                                                                        verbose=True)  # show each iteration or not


# Rigid registration
fil_pc_rigid = 20  # fil_pc_rigid / 100 * (maximum_probability - minimum_probability) + minimum_probability is set as theshhold to filter anchors
slicesl = st_gears.stack_slices_pairwise_rigid([slicesl[i] for i in regis_ilist],
						pili,
						label_col=grp_col,
						fil_pc=fil_pc_rigid,
						filter_by_label=filter_by_label)


# elastic registration
fil_pc_elas = 20  # fil_pc_elas / 100 * (maximum_probability - minimum_probability) + minimum_probability is set as theshhold to filter anchors
pixel_size = 1  # pixel size of elastic field, input a rough average of spots distance here
sigma = 1  # kernel size of Gaussian Filters, with a higher value indicating a smoother elastic field
slicesl = st_gears.stack_slices_pairwise_elas_field([slicesl[i] for i in regis_ilist],
                                                    pili,
                                                    label_col=grp_col,
                                                    pixel_size=pixel_size,
						    fil_pc=fil_pc_elas,
                                                    filter_by_label=filter_by_label,
						    sigma=sigma)
```

### Demo Tutorial
https://github.com/STOmics/ST-GEARS/blob/main/demo.ipynb
Expected run time on a typical quad core Windows is 5 mins.
