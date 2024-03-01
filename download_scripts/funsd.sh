cd $DATASET_DIR

echo "Downloading FUNSD dataset..."
mkdir funsd
cd funsd
wget https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip dataset.zip && rm dataset.zip
