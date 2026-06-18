# HSI Bruises Introduction

Analysis of bruise development and characteristics using Hyperspectral Imaging (HSI) data from the Specim IQ camera.

## Project Overview

This project analyzes hyperspectral image data collected using a **Specim IQ hyperspectral camera** to study bruise formation and progression on different body parts (blood samples, nails) under various conditions (angles, time points).

### Data

The `Data/Raw/` directory contains hyperspectral image datasets organized by timestamp (YYYY-MM-DD_XXX format). Each dataset includes:

- **`capture/`** - Raw hyperspectral image files (.hdr format) from Specim IQ
  - Main hyperspectral image (`*.hdr`)
  - Dark reference (`DARKREF_*.hdr`) for calibration
  - White reference (`WHITEREF_*.hdr`) for normalization
  
- **`metadata/`** - Associated metadata (`.xml` format)
  - Capture parameters and settings
  - Camera and sensor information
  
- **`results/`** - Processed reflectance data
  - Reflectance-corrected images (`REFLECTANCE_*.hdr`)

- **`manifest.xml`** - Dataset manifest and documentation

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository** (if not already cloned):
   ```bash
   git clone <repository-url>
   cd HSI_Bruises_intro
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n hsi-bruises python=3.8
   conda activate hsi-bruises
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Jupyter** (if not already included):
   ```bash
   pippip install jupyter jupyterlab
   ```

### Running Jupyter Notebooks

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   # Or for JupyterLab (more modern interface):
   jupyter lab
   ```

2. **Open a notebook**:
   - Navigate to the repo directory in the Jupyter interface
   - Click on any of the `Data_analysis_*.ipynb` files
   - Start running cells with `Shift + Enter`

## Notebooks Overview

- **`Data_analysis_blood.ipynb`** - Analysis of blood sample hyperspectral data
- **`Data_analysis_Nails.ipynb`** - Analysis of nail sample hyperspectral data
- **`Data_analysis_T0_30deg.ipynb`** - Analysis at baseline (T0) at 30° angle
- **`Data_analysis_T0_60deg.ipynb`** - Analysis at baseline (T0) at 60° angle
- **`Data_analysis_T1_1W_30deg.ipynb`** - Analysis at 1 week (T1) at 30° angle

## Dependencies

| Package | Purpose |
|---------|---------|
| **numpy** | Numerical computing and array operations |
| **matplotlib** | Data visualization and plotting |
| **spectral** | Hyperspectral image processing and analysis |
| **scikit-learn** | Machine learning algorithms |
| **ipywidgets** | Interactive widgets for Jupyter notebooks |

## Data Format

The hyperspectral images are stored in **ENVI format** (.hdr header files):
- `.hdr` - ENVI header file (metadata and structure)
- `.bil`, `.bip`, or `.bsq` - Binary image data (depends on interleave format)

These files can be read using the `spectral` library:
```python
import spectral as spy
img = spy.open_image('path/to/image.hdr')
```

## Workflow

1. Raw hyperspectral images are captured using Specim IQ
2. Data is organized by timestamp in the `Data/Raw/` directory
3. Jupyter notebooks load and process the data
4. Analysis includes:
   - Reflectance correction using dark/white references
   - Spectral analysis and visualization
   - Feature extraction and classification
   - Time-series comparisons across conditions

## File Structure

```
HSI_Bruises_intro/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── Data_analysis_blood.ipynb      # Blood sample analysis
├── Data_analysis_Nails.ipynb      # Nail sample analysis
├── Data_analysis_T0_*.ipynb       # Baseline timepoint analyses
├── Data_analysis_T1_*.ipynb       # Follow-up timepoint analyses
└── Data/
    └── Raw/
        └── [YYYY-MM-DD_XXX]/      # Datasets organized by timestamp
            ├── capture/           # Raw HSI images
            ├── metadata/          # XML metadata
            ├── results/           # Processed results
            └── manifest.xml       # Dataset manifest
```

## Notes

- Ensure your Python environment is activated before running Jupyter
- Large hyperspectral images require sufficient RAM for processing
- Reference images (dark/white) should be captured under the same conditions as sample images for accurate calibration

## License

[Add your license information here]

## Contact

[Add contact information here]
