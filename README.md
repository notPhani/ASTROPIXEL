# AstroPixel_Spectra
AstroPixel_Spctra- Dataset of 3500 galaxies with multi filtered images with respective spectra
A dataset of 3,500 galaxies, each captured in 5 different photometric filters (g, r, i, z, y) along with their corresponding spectra. This dataset is designed for astronomical deep learning, redshift estimation, and galaxy classification.

## 🌌 Dataset Overview
*Total Galaxies: 3,500*

**Filters Used**: g, r, i, z, y (Pan-STARRS1)

**Spectral Data Source**: SDSS DR16

**Size**: 7 GB 

Each galaxy has: ✅ 5 multi-band images (in .fits format)
✅ 1 spectral file (in .csv format with wavelength vs. flux)

##📂 Dataset Structure
```bash
AstroPixel_Spectra/
│── Galaxy_123456/
│   ├── images/
│   │   ├── 123456_g.fits
│   │   ├── 123456_r.fits
│   │   ├── ...
│   ├── spectrum/
│   │   ├── spectrum.csv
│── Galaxy_123457/
│── ...
└── data.csv  # Metadata file with RA, Dec, SpecObjID, etc.
(Each galaxy is stored in its own folder for easy access!)
```
### ⚙️ Data Collection Process
This dataset was generated using an automated script that:
*1️⃣ Queries Pan-STARRS1 (PS1) for galaxy images across 5 filters (g, r, i, z, y).*
*2️⃣ Downloads spectral data from SDSS DR16.*
*3️⃣ Saves each galaxy's images and spectra into structured folders.*
*4️⃣ Uses multi-threading for efficient downloading.*

## Snippet from the Download Script
```python
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits
import requests, os, time, pandas as pd, io

filters = ["g","r","i","z","y"]
def download_galaxy(galaxy_id, vis=False, max_threads=5):
    print(f"🚀 Downloading Galaxy {galaxy_id}")
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(download_filter, filter_band, galaxy_id, vis) for filter_band in filters]
        futures.append(executor.submit(download_spectrum, galaxy_id))
        for future in futures:
            future.result()
    print(f"✅ Finished Galaxy {galaxy_id}")
```
**Full script available in the repository.**

## 📊 How to Use This Dataset
You can load and visualize the FITS images using astropy and matplotlib:

```python
from astropy.io import fits
import matplotlib.pyplot as plt

with fits.open("Galaxy_123456/images/123456_g.fits") as hdul:
    img_data = hdul[0].data
plt.imshow(img_data, cmap="gray")
plt.show()
```
**For spectral data, simply load it as a Pandas DataFrame:**

```python

import pandas as pd
df = pd.read_csv("Galaxy_123456/spectrum/spectrum.csv")
df.plot(x="wavelength", y="flux")
```
## 🔥 Potential Use Cases
*🚀 Astronomical Deep Learning – Train a CNN to reconstruct missing filters*
*🌌 Galaxy Classification – Differentiate between elliptical & spiral galaxies*
*🔭 Redshift Estimation – Use spectral features to predict galaxy distances*

## 📥 Download
The dataset is available on Kaggle:
```python
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "sneakyrat/galaxy-image-filters-with-spectra",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())
```
🛠 Contributors
🔹 Phani Kumar – Data Collection, Processing & Dataset Creation
