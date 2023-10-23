# PopPhy2-AeResNet
Built on top of PopPhy-CNN (Reiman et al., PopPhy-CNN: A Phylogenetic Tree Embedded Architecture for Convolutional Neural Networks to Predict Host Phenotype From Metagenomic Data, 2020). This takes in a microbiome OTU abundance table as an input and uses PopPhy-CNN's algorithm to convert this input into a 2D matrix created by embedding the phylogenetic tree that is populated with the relative abundance of microbial taxa. Then this 2d matrix is passed through an encoder-decoder network to prevent overfitting, then outputs from the decoder network are fed through a CNN architecture skip connections, then outputs from the CNN architecture are fed through a fully connected neural network which outputs the phenotype prediction.


## Instructions

## Data Preprocessing

Run data-preproccessing/SKG1_ML.Rmd <br>
-change "SKG2021.Rdata" to specific Rdata file <br>
-this script takes a Rdata file and uses Phyloseq to exports a "otu.csv", "sample_data.csv", and "taxa.csv" into the local directory

Run preprocessing.ipynb <br>
-change "Get labels" section to get 0 and 1 labels for desired specificed phenotype <br>
-change "folder name" variable in "export dataframes to tsv and txt files" to get data export <br>
-this script assigns a label for each mouse and exports "abundance.tsv" and corresponding "labels.txt" into specificied directory in data/YOURDATASET folder 

## Modify ML Model Architechture
-Modify Architechture or Parameters of ML models in src/model/ (src/model/PopPhy2.py is default and highest accuracy)

## Train ML models
-configure model and dataset in "Configuration" section in src/train.ipynb and run <br>
-note: data/YOURDATASET/PopPhy-tree-core.pkl needs to be deleted before you change the threshold in src/train.ipynb



## Citation:
* Reiman D, Metwally AA, Sun J, Dai Y. PopPhy-CNN: A Phylogenetic Tree Embedded Architecture for Convolutional Neural Networks to Predict Host Phenotype From Metagenomic Data. IEEE J Biomed Health Inform. 2020 Oct;24(10):2993-3001. doi: 10.1109/JBHI.2020.2993761. Epub 2020 May 11. PMID: 32396115. [[paper](https://pubmed.ncbi.nlm.nih.gov/32396115/)]

