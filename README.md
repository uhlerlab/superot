# superot

This is the code repository for the paper https://arxiv.org/pdf/2007.12098.pdf. The dataset can be obtained from Experiment 1 in https://github.com/AllonKleinLab/paper-data/blob/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation/README.md. 

Here is a description of each of the files. For `data_preprocess.py`, `main_conditional.py`, `main.py`, and `main_supervised.py`, please change COUNTS_MATRIX, CLONE_ANNOTATION, CELL_METADATA to the corresponding file names in your directory. 


1. `GAN.py` --> GAN model code 
2. `cGAN.py` --> Conditional GAN model code
3. `data_preprocess.py` --> Code for setting up the data loaders in the unsupervised, semi-supervised, and supervised settings.
4. `utils.py` --> Code for parsing arguments / data loaders
5. `main_conditional.py` --> Code for training and evaluating the conditional GAN in the unsupervised setting. 
6. `main.py` --> Code for training and evaluating the GAN in the semi-supervised setting. Note that, depending on the number of supervised points you choose to use, you will have to change 'num_points' accordingly. 
7. `main_supervised.py` --> Code for training and evaluating the GAN in the supervised setting. 

To use, first save the necessary data loaders by running `data_preprocess.py`, being sure to download all the data and have it saved in the same directory first. Then, create an empty folder named 'results' and start up a new visdom server by running `python -m visdom.server -port 3000` in terminal. In a new tab, run either `main_conditional.py`, `main.py` or `main_supervised.py` to obtain your results. 
