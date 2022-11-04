## Exploring the Use of Protein Language Models to Predict Intrinsic Disorder in Proteins
## Supplementary code for master thesis by Felix Gabler

### Abstract excerpt
Intrinsically disordered proteins (IDPs) are an abundant class of proteins that do not adopt a fixed or ordered three-dimensional structure,
typically in the absence of molecular interactions with other proteins or macromolecules such as DNA or RNA.
IDPs may be completely unstructured or partially structured with sections of intrinsically disordered regions (IDRs).
Since these proteins are associated with various diseases such as Alzheimer's disease and Huntington's disease, their study is of great importance.
Unfortunately, experimental determination of intrinsic disorder in proteins is tedious and expensive, as evidenced by the lack of data in disorder databases.
While there are many computational methods for predicting IDRs, most are rather inefficient and therefore not suitable for larger databases.
In our work, we evaluated the use of transformer-based protein language models (pLMs) for the prediction of intrinsic disorder in proteins.
These deep learning models are trained in an unsupervised manner exclusively on sequences and have been shown to extract biophysical properties of amino acids in their resulting embeddings.
To evaluate the utility of these models for our use case, we have experimented with various state-of-the-art, pre-trained pLMs,
the degree of fine-tuning useful, and the complexity of the models required to extract disorder-related information from the embeddings.
In addition, we explored the benefits of training with more nuanced continuous disorder scores.

### About this repository

This repository contains all the data (see folder `/data`), methods (see folder `/bin/disorder`) and experiments (see folder `/experiments` and other folders as described) described in the thesis.
