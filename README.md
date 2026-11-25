<h1 align="center">Filter WMC Datatype</h1>

<p align="center">
  Removes from each tract of a white matter classification (WMC) the streamlines rejected by tractogram filtering such as SIFT or COMMIT, using a binary streamline label file.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.621"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.621-blue.svg" alt="Run on brainlife.io"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Removes from each tract of a white matter classification (WMC) the streamlines rejected by tractogram filtering such as SIFT or COMMIT, using a binary streamline label file.**

Multiplies the WMC streamline assignments by a binary keep/remove label per streamline of the original tractogram, so tracts keep only the streamlines that survived filtering, and regenerates the per-tract JSON files for visualisation. Use it after running a tractogram filtering algorithm on a tractogram that was already segmented. The labels must index the same tractogram used to build the WMC.

| | |
|---|---|
| **Inputs** | `neuro/wmc` · `neuro/labels` · `neuro/track/tck` |
| **Outputs** | `neuro/wmc` |
| **Container** | `brainlife/dipy:1.4.1` |
| **Resources** | 8 CPUs · 40 GB memory · 3 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.621](https://doi.org/10.25663/brainlife.app.621) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `wmc` | `neuro/wmc` | yes | White matter classification (classification.mat, tract JSONs and surfaces) built on the tractogram. |
| `labels` | `neuro/labels` (binary) | yes | Binary label per streamline of the tractogram (index.csv): 1 keeps it, 0 removes it. |
| `track` | `neuro/track/tck` | yes | Needed to generate files for visualization purposes |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `wmc` | `neuro/wmc` | Filtered white matter classification with updated per-tract visualisation files. |

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.621](https://doi.org/10.25663/brainlife.app.621), choose `neuro/wmc`, `neuro/labels`, `neuro/track/tck` from your project and press **Execute**. The results are saved to your project as `neuro/wmc`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `brainlife/dipy:1.4.1`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "classification": "/path/to/wmc/classification",
       "tracts": "/path/to/wmc/tracts",
       "surfaces": "/path/to/wmc/surfaces",
       "index": "/path/to/labels/index",
       "track": "/path/to/track/track"
   }
   ```

3. Run `./main`.

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- Filter WMC Datatype. brainlife.io App. [doi:10.25663/brainlife.app.621](https://doi.org/10.25663/brainlife.app.621)
- Garyfallidis E, et al. DIPY, a library for the analysis of diffusion MRI data. Front Neuroinform 8:8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S, Descoteaux M, Nimmo-Smith I and Dipy Contributors (2014). DIPY, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics, vol.8, no.8.
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Brad Caron

**Contributors**

- Soichi Hayashi

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io, which runs this App, is publicly funded by Wellcome, the U.S. National Institutes of Health and the U.S. National Science Foundation; acknowledging that support helps keep it running.</sub>
