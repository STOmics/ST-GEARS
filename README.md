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

uniform_weight = False  # not to use uniform weight, instead, use Distributive Constraints
filter_by_label = True  # Filter groups of spot that do not co-occur in two sections when computing anchors

warnings.filterwarnings('ignore')

# Compute anchors
pili, tyscoreli, alphali, regis_ilist, ali, bli = st_gears.serial_align(slicesl, anncell_cid, label_col='annotation',
                                                                        start_i=0, end_i=len(slicesl)-1,
                                                                        tune_alpha_li=[0.8, 0.2, 0.05, 0.013],
                                                                        numItermax=150,
                                                                        uniform_weight=uniform_weight,
                                                                        filter_by_label=filter_by_label,
                                                                        verbose=True)
# Rigid registration
slicesl = st_gears.stack_slices_pairwise_rigid([slicesl[i] for i in regis_ilist],
						pili,
						label_col='annotation',
						fil_pc=20,
						filter_by_label=filter_by_label)

# elastic registration
pixel_size = 1  # pixel size of elastic field, input a rough average of spots distance here
sigma = 1  # kernel size of Gaussian Filters, with a higher value indicating a smoother elastic field
slicesl = st_gears.stack_slices_pairwise_elas_field([slicesl[i] for i in regis_ilist],
                                                    pili,
                                                    label_col='annotation',
                                                    pixel_size=pixel_size,
						    fil_pc=20,
                                                    filter_by_label=filter_by_label,
						    sigma=sigma)
```

