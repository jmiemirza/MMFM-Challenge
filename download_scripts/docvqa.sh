cd $DATASET_DIR

echo "Downloading DocVQA dataset..."
mkdir docvqa
cd docvqa
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/DocVQA.tar.gz
tar xvf DocVQA.tar.gz && rm DocVQA.tar.gz
mkdir jpgs
cd jpgs
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/pdfs/DocVQA.tar.gz # download the pdfs
tar xvf DocVQA.tar.gz && rm DocVQA.tar.gz
mv DocVQA/* . && rm -r DocVQA
echo "Converting pdfs to jpgs... (this may take a while)"
for file in *.pdf; do basename="${file%.pdf}"; pdftoppm -jpeg -r 150 "$file" "$basename" && mv "$basename-1.jpg" "$basename.jpg" && rm "$file"; done
