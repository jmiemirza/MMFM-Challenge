cd $DATASET_DIR

echo "Downloading WTQ dataset..."
mkdir wtq
cd wtq
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/WikiTableQuestions.tar.gz
tar xvf WikiTableQuestions.tar.gz && rm WikiTableQuestions.tar.gz
mkdir jpgs
cd jpgs
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/pdfs/WikiTableQuestions.tar.gz # download the pdfs
tar xvf WikiTableQuestions.tar.gz && rm WikiTableQuestions.tar.gz
mv WikiTableQuestions/* . && rm -r WikiTableQuestions
echo "Converting pdfs to jpgs... (this may take a while)"
for file in *.pdf; do basename="${file%.pdf}"; pdftoppm -jpeg -r 150 "$file" "$basename" && mv "$basename-1.jpg" "$basename.jpg" && rm "$file"; done
