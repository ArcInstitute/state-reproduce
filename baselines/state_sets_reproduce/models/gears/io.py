"""
Reference: parts of https://github.com/S-C-I/GEARS/blob/main/gears/utils.py
"""

import os
import logging
import requests
from zipfile import ZipFile
import tarfile
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """

    if os.path.exists(save_path):
        logger.info("Found local copy...")
    else:
        logger.info("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()


def zip_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for zip file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset
    """

    if os.path.exists(save_path):
        logger.info("Found local copy...")
    else:
        dataverse_download(url, save_path + ".zip")
        logger.info("Extracting zip file...")
        with ZipFile((save_path + ".zip"), "r") as zip:
            zip.extractall(path=data_path)
        logger.info("Done!")


def tar_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for tar file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset

    """

    if os.path.exists(save_path):
        logger.info("Found local copy...")
    else:
        dataverse_download(url, save_path + ".tar.gz")
        logger.info("Extracting tar file...")
        with tarfile.open(save_path + ".tar.gz") as tar:
            tar.extractall(path=data_path)
        logger.info("Done!")



def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """

    if os.path.exists(save_path):
        logger.info("Found local copy...")
    else:
        logger.info("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()