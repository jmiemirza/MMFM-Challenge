cd $DATASET_DIR

echo "Downloading InfoVQA dataset..."
mkdir infovqa
cd infovqa
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/InfographicsVQA.tar.gz
tar xvf InfographicsVQA.tar.gz && rm InfographicsVQA.tar.gz
mkdir jpgs
cd jpgs
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/pdfs/InfographicsVQA.tar.gz # download the pdfs
tar xvf InfographicsVQA.tar.gz && rm InfographicsVQA.tar.gz
mv InfographicsVQA/* . && rm -r InfographicsVQA
echo "Converting pdfs to jpgs... (this may take a while)"
for file in *.pdf; do basename="${file%.pdf}"; pdftoppm -jpeg -r 150 "$file" "$basename" && mv "$basename-1.jpg" "$basename.jpg" && rm "$file"; done
