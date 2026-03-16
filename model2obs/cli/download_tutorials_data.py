#!/usr/bin/env python3

## @file download_db.py
#
#
## @author Enrico Milanese <enrico.milanese@whoi.edu>
#
## @date Thu 27 Feb 2025

##########################################################################
import argparse
import os
import requests
from tqdm import tqdm
import zipfile
##########################################################################

#------------------------------------------------------------------------------#
def download_zenodo_record(zenodo_id, download_to):
    """Download a zenodo record given its id and save it locally.

    Args:
    zenodo_id (int)       --  ID of Zenodo record to download
    download_to (str)     -- folder to download files to

    Returns:
    filenames     --    list of filenames downloaded
    """

    zenodo_urls, zenodo_filenames = fetch_zenodo_url(zenodo_id)

    for url, fname in zip(zenodo_urls, zenodo_filenames):
        fname = download_to + fname
        print(
            f"    Downloading {url} to {fname}..."
        )
        download_file(url, fname)

    return zenodo_filenames

#------------------------------------------------------------------------------#
def fetch_zenodo_url(zenodo_id):
    """Get URL from zenodo_id.

    Args:
    zenodo_id (int)       --  ID of Zenodo record to download

    Returns:
    file_urls    --  urls to download record files from
    filenames    --  filenames from record
    """

    metadata_url = f"https://zenodo.org/api/records/{zenodo_id}"
    r = requests.get(metadata_url)
    r.raise_for_status()
    metadata = r.json()

    files = metadata['files']
    filenames = []
    file_urls = []

    for f in files:
        filenames.append(f['key'])
        file_urls.append(f['links']['self'])

    return file_urls, filenames        

#------------------------------------------------------------------------------#
def download_file(url, local_filename):
    """Download a file from a URL and save it locally.

    Args:
    url (str)             --  URL of the file to download.
    local_filename (str)  --  Local filename to save the file to.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_filename, "wb") as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

#------------------------------------------------------------------------------#
def unzip_file(zip_filepath, extract_to):
    """Unzip a file to a specified directory.

    Args:
    zip_filepath (str)  --  Path to the zip file to extract.
    extract_to (str)    --  Directory to extract the zip file to.
    """

    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        members = zip_ref.infolist()
        total_size = sum(m.file_size for m in members)  # Total archive size (uncompressed)
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Extracting') as pbar:
            for member in members:
                dest_path = os.path.join(extract_to, member.filename)
                if member.is_dir():
                    os.makedirs(dest_path, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with zip_ref.open(member) as source, open(dest_path, 'wb') as target:
                    while True:
                        chunk = source.read(1024 * 1024)  # 1MB at a time
                        if not chunk:
                            break
                        target.write(chunk)
                        pbar.update(len(chunk))

#------------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser(description='Script to download the datasets used in the tutorials for the 2025 Crocodile workshop.')
    parser.add_argument('--destination', type=str, help="Destination folder to download data to (default: ./2025_Crocodile_tutorial/)", required=False, default="./2025_Crocodile_tutorial/")

    args = parser.parse_args()
    destination = args.destination
    if not destination[-1] == "/":
        destination += "/"
    os.makedirs(destination, exist_ok=True)

    # Download the dataset
    print(f"Downloading tutorials data...")
    zenodo_id = 17524900
    filenames = download_zenodo_record(zenodo_id, destination)

    # Unzip the dataset
    for fname in filenames:
        if fname.endswith(".zip"):
            fname = destination + fname
            print(f"Unzipping {fname} to {destination}...")
            unzip_file(fname, destination)
            print(f"Removing zip archive {fname}...")
            os.remove(fname)

    print("Download complete.")

##########################################################################
if __name__ == "__main__":
    main()
