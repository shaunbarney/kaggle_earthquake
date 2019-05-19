"""Script to download kaggle dataset"""
import os
import kaggle


kaggle.api.authenticate()

if not os.path.exists('./dataset'):
    print("Making dataset dir")
    os.mkdir('./dataset')
comp_name = 'LANL-Earthquake-Prediction'
print(f"Downloading {comp_name}")
kaggle.api.competition_download_files(comp_name, path='./dataset/')
