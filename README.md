## Intro
This directory contains quick-and-easy verification codes for the work FSL-Rectifier. 

## Test FSL model
* Before running the test-time augmentation, please download the test dataset and trained FSL models from this [Google Drive Link](https://drive.google.com/file/d/1NzYCUdd0Zmr3Ogp-GJWo_aVLeslkWmw_/view?usp=drive_link).
* Please adjust `animals.yaml` for the correct dataset folder location.
* Please adjust and run `python auto_experiment.py` to test the trained FSL models, which saves all results under folder `./outputs` as `.txt` files.

## Acknowledgements
* Please note that our codes are built based on [FEAT](https://github.com/Sha-Lab/FEAT) and [FUNIT](https://github.com/NVlabs/FUNIT).
