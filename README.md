# ML_Project

## *Epigenetic plasticity as a signature for memory allocation -:computer: Miniproject BIO-322*

## Aim

In this project, we will be analyzing high-dimensional data of Giulia Santoni, of gene expression levels in multiple cells of a mouse brain under three different experimental conditions(that we call KAT5, CBP and eGFP). Our goal is to use machine learning techniques to predict, for each cell, the experimental condition under which it was measured based on its gene expression levels.  

## :handshake: Contributors

Guillaume Belissent and Camille Challier

This is a private project but we accept advices on how to improve our project and optimize the code!

## Project description

The training data consist of normalized counts for 32285 genes in 5000 different cells together with the experimental condition under which each cell was measured.  The test data contains only the normalized counts and our task is to predict the experimental condition.
In this project we tried to find the best feature selection and machine learning techniques to predict the experimental condition.
Through this work, we hope to gain new insights into the markers of epigenetic modifications in different experimental conditions.

![Hopfiel Network diagram](https://upload.wikimedia.org/wikipedia/commons/b/b4/Hopfield%27s_net.png)

## :open_book: Requirements and Installation

- This project is available on [GitHub Pages](https://github.com/) with the following link: https://github.com/gbelis/ML_Project.git
  To install the project from GitHub:
  ```
  $ git clone https://github.com/gbelis/ML_Project.git
  $ cd ../path/to/the/project
  ```
- Programming language: Julia and one python notebook
- Working platform: Visual Studio Code
-  Julia >= v1.7.3
- Libraries: 
  - MLCourse package available on https://github.com/jbrea/MLCourse.git
  - PlotlyJS

## :card_file_box: Files and contents

**Structure:**

- |-- Best_Results.................................................................................................................................1
     - |-- LassoClassifier.jl............................................................................................................................1.1

- |-- Features_selection..............................................................................................................................2
     - |-- pca_tuning.jl................................................................................................................................2.1
     - |-- statistics.jl....................................................................................................................................2.2
     - |-- features_engeneering_tuning.jl ..................................................................................................2.3

- |-- Models.................................................................................................................................................3
     - |-- LassoClassifier.jl............................................................................................................................3.1
     - |-- MultinomialClassifier.jl...................................................................................................................3.2
     - |-- NN.jl...............................................................................................................................................3.3
     - |-- NNClassifier.jl.................................................................................................................................3.4    
     - |-- RandomForestClassifier.jl..............................................................................................................3.5    
     - |-- RidgeClassifier.jl............................................................................................................................3.6
     - |-- SVC.jl.............................................................................................................................................3.7
     - |-- SplitClassification.jl........................................................................................................................3.8
     - |-- Supervised_UMAP.ipynb...............................................................................................................3.9
     - |-- XGB.jl............................................................................................................................................3.10
     
- |-- Plots ......................................................................................................................................................4
     - |-- Features_selection_plot.html..........................................................................................................4.1
     - |-- corr_plot_norm.png........................................................................................................................4.2
     - |-- histogram.html................................................................................................................................4.3
     - |-- pca_plot.html..................................................................................................................................4.4
     - |-- variance_plot.png...........................................................................................................................4.5

- |-- Submission............................................................................................................................................5
     - |-- 
     
- |-- data ......................................................................................................................................................6
     - |-- indexes.csv....................................................................................................................................6.1
     - |-- test.csv.gz.....................................................................................................................................6.2
     - |-- train.csv.gz....................................................................................................................................6.3

     - |-- README.md ...............................................................................................................................7
     - |-- Visualisations.jl.............................................................................................................................8
     - |-- cross_val.jl....................................................................................................................................9
     - |-- data_processing.jl.........................................................................................................................10
     - |-- models.jl......................................................................................................................................11

**Contents :**

2. Best Models obtained

     1.1 Best linear mathod : Lasso Classifier tuning 

     1.2 Best non-linear method :

2. Methods tested and used for genes selection and dimensionality reduction.

    2.1. PCA tuning and evaluation of the accuracy after a PCA.

    2.2. Selection of more important genes using a statistical T-test.

    2.3. Testing of data processing techniques.


3. Different machine learning models that we have used to predict the experimental condition.

    3.1. Tuning of the lambda hyperparameter of a Lasso Classifier and best model found.

    3.2. Simple Mutlinomial Classifier on all the training data.

    3.3. Neural Network tuning with mean difference features selection and PCA.

    3.4. Neural Network tuning with T-test features selection and PCA.

    3.5. Tuning of n_trees and max_depth hyperparameters of a Random Forest Classifier.

    3.6. Tuning of a Ridge Classifier.

    3.7. Tuned Machine of a SVC model.

    3.8. Classification with two different steps. The first predicting wether the cell is control (eGFP) or modified (CBP or KAT5). Then split KAT5 and CBP after. 

    3.9. Supervised UMAP in python.

    3.10. Tuning of XGBoost Model.

4. Intersting Plots useful for data interpretation, data analysis and features selection.

    4.1. Plot of the accuracy variation with different features selection techniques ( call-rates, mean variation and t-test).

    4.2. Correlation plot between the four more important genes, according to a t-test. With a previous standardisation of the data.

    4.3. Histogramme showing the distributions of the experimental conditions in the training data.

    4.4. Spatial representation of the labels repartition using the estimated 3 more important genes.

    4.5. PCA explained variance plot


5. Best Kaggle Submissions.

6. Train, Test data and indexes of selected genes after cleaning.

7. Readme.

8. Code for all figures.

9. Template for feature selection tuning with a certain model.

10. Functions used for data processing and features engeneering.

11. Functions for evaluation and cross validation of machine learning techniques.

