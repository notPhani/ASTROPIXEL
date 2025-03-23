import requests
import os
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import io
import time


filters = ["g","r","i","z","y"]

#filename_api = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters={filter}"
#cutout_api = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red={filename}&format=fits&x={ra}&y={dec}&size=288&wcs=1&imagename={file_name}"

import requests

def get_filename(ra, dec, filter_band):
    url = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters={filter_band}"
    response = requests.get(url)

    if response.status_code == 200:
        lines = response.text.strip().split("\n")
        if len(lines) > 1:
            columns = lines[1].split()  # Extract second line (data row)
            if len(columns) >= 8:
                filename = columns[7] 
                file_name = columns[8] # Extracting the full filename path
                return filename,file_name
        print(f"No valid filename found for {ra}, {dec} in {filter_band}")
        return None
    else:
        print(f"Failed to get filename (Status {response.status_code})")
        return None

def download_filter(filter_band,galaxy_id,vis = True):
    base_dir = "D:/Dataset"
    data = "Data/Galaxy_ids/data.csv"
    df = pd.read_csv(data)
    row = df[df["specobj_id"] == f"'{galaxy_id}'"]
    ra,dec = row.iloc[0]['ra'],row.iloc[0]['dec']
    filename, file_name = get_filename(ra,dec,filter_band)
    base_url = f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red={filename}&format=fits&x={ra}&y={dec}&size=288&wcs=1&imagename={file_name}"
    galaxy_folder = os.path.join(base_dir,f"Galaxy_{int(galaxy_id.replace("'",""))}",'images')
    os.makedirs(galaxy_folder,exist_ok=True)
    save_path = os.path.join(galaxy_folder,f"{int(galaxy_id.replace("'",""))}_{filter_band}.fits")

    response = requests.get(base_url,stream=True)
    if response.status_code == 200:
        with open(save_path,"wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
            print(f"Downloaded {save_path}")


    if vis:
        with fits.open(save_path) as hdul:
            image_data = hdul[0].data
        plt.imshow(image_data,cmap="gray")
        plt.show()

def download_spectrum(spec_id,save_csv=True):
    save_dir = "D:/Dataset" 
    data = "Data/Galaxy_ids/data.csv"
    df = pd.read_csv(data)
    row = df[df["specobj_id"] == f"'{spec_id}'"]  # Ensure type matches
    
    if row.empty:
        print(f"❌ SpecObjID {spec_id} not found in CSV.")
        return None
    
    # Extract Plate, MJD, FiberID (Ensure proper formatting)
    plate = f"{int(str(row.iloc[0]['#plate']).replace("'", "").strip()):04d}" # Plate is usually fine
    mjd = f"{int(str(row.iloc[0]['mjd']).replace("'", "").strip()):04d}" # MJD is usually fine
    fiberID = f"{int(str(row.iloc[0]['fiberid']).replace("'", "").strip()):04d}"
    spectrum_url = f"https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/{plate}/spec-{plate}-{mjd}-{fiberID}.fits"
    response = requests.get(spectrum_url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download spectrum for SpecObjID {spec_id}")
        return None

    with fits.open(io.BytesIO(response.content)) as hdul:
        data = hdul[1].data  # Extract spectrum data
        wavelengths = 10 ** data["loglam"]  # Convert loglam to wavelength
        flux = data["flux"]
    spectrum_df = pd.DataFrame({"wavelength": wavelengths, "flux": flux})
    
    if save_csv:
        save_path = f"{save_dir}/Galaxy_{spec_id}/spectrum/spectrum.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        spectrum_df.to_csv(save_path, index=False)
        print(f"Spectrum data saved: {save_path}")

    return spectrum_df

def download_galaxy(galaxy_id, vis=False, max_threads=5):
    print(f"🚀 Starting download for Galaxy {galaxy_id}")
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(download_filter, filter_band, galaxy_id, vis) for filter_band in filters]
        futures.append(executor.submit(download_spectrum, galaxy_id))

        for future in futures:
            future.result()  # Wait for individual tasks to complete
    
    print(f"✅ Finished Galaxy {galaxy_id}")


def get_galaxy_batches(csv_path, batch_size=50):
    df = pd.read_csv(csv_path)
    galaxy_ids = df["specobj_id"].astype(str).str.replace("'", "") # Remove quotes if present
    return galaxy_ids

# Example Usage
csv_path = "Data/Galaxy_ids/data.csv"
batches = get_galaxy_batches(csv_path, batch_size=50)[475:]
for galaxy in batches:
    start = time.time()
    download_galaxy(galaxy)
    end = time.time()
    print(galaxy, end-start)

