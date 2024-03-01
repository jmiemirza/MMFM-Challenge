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
cd ../..

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
cd ../..


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
cd ../..


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
cd ../..

