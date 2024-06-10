cd $DATASET_DIR

echo "Downloading TabFact dataset..."
mkdir tabfact
cd tabfact
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/TabFact.tar.gz
tar xvf TabFact.tar.gz && rm TabFact.tar.gz
mkdir jpgs
cd jpgs
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/pdfs/TabFact.tar.gz # download the pdfs
tar xvf TabFact.tar.gz && rm TabFact.tar.gz
mv TabFact/* . && rm -r TabFact
echo "Converting pdfs to jpgs... (this may take a while)"
for file in *.pdf; do basename="${file%.pdf}"; pdftoppm -jpeg -r 150 "$file" "$basename" && mv "$basename-1.jpg" "$basename.jpg" && rm "$file"; done
