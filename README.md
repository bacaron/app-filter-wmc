[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-brainlife.app.619-blue.svg)](https://doi.org/10.25663/brainlife.app.619)

# Filter WMC Datatype

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Removes from each tract of a white matter classification (WMC) the streamlines rejected by tractogram filtering such as SIFT or COMMIT, using a binary streamline label file.**

Multiplies the WMC streamline assignments by a binary keep/remove label per streamline of the original tractogram, so tracts keep only the streamlines that survived filtering, and regenerates the per-tract JSON files for visualisation. Use it after running a tractogram filtering algorithm on a tractogram that was already segmented. The labels must index the same tractogram used to build the WMC.

| | |
|---|---|
| 📥 **Inputs** | `neuro/wmc` · `neuro/labels` · `neuro/track/tck` |
| 📤 **Outputs** | `neuro/wmc` |
| 🐳 **Container** | `brainlife/dipy:1.4.1` |
| ⚙️ **Resources** | 8 CPUs · 40 GB memory · 3 h walltime |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.621](https://doi.org/10.25663/brainlife.app.621) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `wmc` | `neuro/wmc` | yes | White matter classification (classification.mat, tract JSONs and surfaces) built on the tractogram. |
| `labels` | `neuro/labels` (binary) | yes | Binary label per streamline of the tractogram (index.csv): 1 keeps it, 0 removes it. |
| `track` | `neuro/track/tck` | yes | Needed to generate files for visualization purposes |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `wmc` | `neuro/wmc` | Filtered white matter classification with updated per-tract visualisation files. |

📚 **References used by this App:**
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)
- Garyfallidis E, et al. DIPY, a library for the analysis of diffusion MRI data. Front Neuroinform 8:8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)

<!-- brainlife-app:end -->

This app will filter streamline indices found in the neuro/wmc datatype based on a labels datatype containing binary indices for the input tractogram used to generate the neuro/wmc datatype. This app should be performed in cases where a tractogram filtering algorithm like SIFT or COMMIT was used following the segmentation of white matter tracts. The goal is to prune out the streamlines from each tract that didn't survive those tractogram filtering algorithms.

### Authors

- Brad Caron (bacaron@utexas.edu)

### Contributors

- Soichi Hayashi (shayashi@iu.edu)

### Funding Acknowledgement

brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your publications and code reusing this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)

### Citations

We kindly ask that you cite the following articles when publishing papers and code using this code.

1. Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). https://doi.org/10.1038/s41597-019-0073-y

2. Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S, Descoteaux M, Nimmo-Smith I and Dipy Contributors (2014). DIPY, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics, vol.8, no.8.

#### MIT Copyright (c) 2020 brainlife.io The University of Texas at Austin and Indiana University

## Running the App

### On Brainlife.io

You can submit this App online at [https://doi.org/10.25663/brainlife.app.619](https://doi.org/10.25663/brainlife.app.619) via the 'Execute' tab.

### Running Locally (on your machine)

1. git clone this repo

2. Inside the cloned directory, create `config.json` with something like the following content with paths to your input files.

```json
{
    "track": "/input/track/track.tck",
    "index": "/input/labels/index.csv",
    "classification": "/input/wmc/classification.mat",
    "surfaces": "/input/wmc/surfaces",
    "tracts": "/input/wmc/tracts"
}
```

### Sample Datasets

You can download sample datasets from Brainlife using [Brainlife CLI](https://github.com/brain-life/cli).

```
npm install -g brainlife
bl login
mkdir input
bl dataset download
```

3. Launch the App by executing 'main'

```bash
./main
```

## Output

The main output of this App is a filtered wmc datatype.

#### Product.json

The secondary output of this app is `product.json`. This file allows web interfaces, DB and API calls on the results of the processing.

### Dependencies

This App only requires [singularity](https://www.sylabs.io/singularity/) to run. If you don't have singularity, you will need to install following dependencies.

- Dipy: https://dipy.org/
- nibabel: https://nipy.org/nibabel/
- numpy: https://numpy.org/
- pandas: https://pandas.pydata.org/
- scipy: https://scipy.org/

#### MIT Copyright (c) 2020 brainlife.io The University of Texas at Austin and Indiana University
