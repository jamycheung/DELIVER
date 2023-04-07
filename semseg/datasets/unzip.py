import zipfile

with zipfile.ZipFile("data/MCubeS/multimodal_dataset.zip", "r") as zip_ref:
    for name in zip_ref.namelist():
        try:
            zip_ref.extract(name, "multimodal_dataset_extracted/")
        except zipfile.BadZipFile as e:
            print(e)