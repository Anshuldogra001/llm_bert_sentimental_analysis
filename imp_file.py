import gdown
import os


def download_file_from_drive(destination_folder):
    download_link = 'https://drive.google.com/uc?id=1s_Ps2DRwG9x635H0b1oX1a6puP_iIazM'
    destination_path = os.path.join(destination_folder, f'pytorch_model.bin') 
    gdown.download(download_link, destination_path, quiet=False)

    print(f"File downloaded to {destination_path}")
destination_folder = 'saved_model'

download_file_from_drive(destination_folder)
