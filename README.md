# ST-GEARS
A Spatial Transcriptomics Geospatial Profile Recovery Tool through Anchors
![flow](https://github.com/STOmics/ST-GEARS/assets/96898334/6785a509-8b57-43f5-ba19-162ecad7ed1b)

### Methods
`serial_align` computes  probabilistic anchors between adjacent sections in serial, using Optimal Transport with Distributive Constraints, based on both expression and coordinates data

`stack_slices_pairwise_rigid` rigidly aligns sections using Procrustes Analysis

`stack_slices_pairwise_elas_field` eliminates distorsions through Gaussian Smoothed Elastic Fields. Validity is proved mathematically

`plot_scatter_by_grid` of class `RegisPlotter` plots registrated sections as well as intermediate results, such as no. anchors distribution and constraints distribution

### Installation
```python
git clone https://github.com/STOmics/ST-GEARS.git

cd ST-GEARS

pip install .

pip install -r requirements.txt
```

### Input and Output
To use ST-GEARS to recover geospatial profile in 3D, you need both expressional and structural information, structured in a list of anndata.Anndata. Rigid and Elastic registration results will be added to .obsm by ST-GEARS:
![dataformat](https://github.com/STOmics/ST-GEARS/assets/96898334/3db7a908-22db-42d5-bbdf-8f22ebb689e4)

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
                                                                        start_i=0, end_i=len(slicesl)-1,
                                                                        tune_alpha_li=[0.8, 0.2, 0.05, 0.013],  # regularization factor list, recommend to fill values exponentially change among 0 and 1. Higher number indicates finer tuning resolution
                                                                        numItermax=150, 
                                                                        uniform_weight=uniform_weight,
                                                                        filter_by_label=filter_by_label,
                                                                        verbose=True)


# Rigid registration
fil_pc_rigid = 20  # fil_pc / 100 * (maximum_probability - minimum_probability) + minimum_probability is set as theshhold to filter anchors

slicesl = st_gears.stack_slices_pairwise_rigid([slicesl[i] for i in regis_ilist],
						pili,
						label_col=grp_col,
						fil_pc=fil_pc_rigid,
						filter_by_label=filter_by_label)


# elastic registration
fil_pc_elas = 20  # fil_pc / 100 * (maximum_probability - minimum_probability) + minimum_probability is set as theshhold to filter anchors
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

