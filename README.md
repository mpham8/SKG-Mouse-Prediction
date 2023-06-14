# PopPhy2-AeResNet

Built on top of PopPhy-CNN (Reiman et al., PopPhy-CNN: A Phylogenetic Tree Embedded Architecture for Convolutional Neural Networks to Predict Host Phenotype From Metagenomic Data, 2020). This takes in a microbiome OTU abundance table as an input and uses PopPhy-CNN's algorithm to convert this input into a 2D matrix created by embedding the phylogenetic tree that is populated with the relative abundance of microbial taxa. Then this 2d matrix is passed through an encoder-decoder network to prevent overfitting, then outputs from the decoder network are fed through a CNN architecture skip connections, then outputs from the CNN architecture are fed through a fully connected neural network which outputs the phenotype prediction.


## Citation:
* Reiman D, Metwally AA, Sun J, Dai Y. PopPhy-CNN: A Phylogenetic Tree Embedded Architecture for Convolutional Neural Networks to Predict Host Phenotype From Metagenomic Data. IEEE J Biomed Health Inform. 2020 Oct;24(10):2993-3001. doi: 10.1109/JBHI.2020.2993761. Epub 2020 May 11. PMID: 32396115. [[paper](https://pubmed.ncbi.nlm.nih.gov/32396115/)]

## Instructions
0) (Optional preprocessing) Add raw .Rdata file into /data-preprocessing and adjust/run /data-preprocessing/SKG1_ML.Rmd to get "abundance.tsv" file output and adjust/run /data-preprocessing/preprocessing.ipynb to get "labels.txt" file output
1) Add datasets into into a new directory in /data directory: abundance table as "abundance.tsv" and corresponding binary labels as "labels.txt" 
2) Adjust hyperparameters in /src/models/PopPhy.py
3) Get model accuracy in /src/train.ipynb or Make predictions in /src/make_pred.ipynb