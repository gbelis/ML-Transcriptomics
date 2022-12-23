# ML_Project

## *Epigenetic plasticity as a signature for memory allocation - :computer: Miniproject BIO-322*

## Aim

In this project, we will be analyzing high-dimensional data of Giulia Santoni, of gene expression levels in multiple cells of a mouse brain under three different experimental conditions(that we call KAT5, CBP and eGFP). Our goal is to use machine learning techniques to predict, for each cell, the experimental condition under which it was measured based on its gene expression levels.  

## :handshake: Contributors

Guillaume Belissent and Camille Challier

## Project description

The training data consists of normalized counts for 32285 genes in 5000 different cells together with the experimental condition under which each cell was measured. The test data contains only the normalized counts and our task is to predict the experimental condition.
In this project we tried to find the best feature selection and machine learning techniques to predict the experimental condition.
Through this work, we hope to gain new insights into the markers of epigenetic modifications in different experimental conditions.

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

- |-- Best_Results......................................................................................................................................1
     - |-- LassoClassifier.jl..........................................................................................................................1.1

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
     - |-- histogram.html.................................................................................................................................4.1
     - |-- UMAP_plot.png .............................................................................................................................4.2
     - |-- clustermap.png..............................................................................................................................4.3
     - |-- corr_plot_norm.png........................................................................................................................4.4
     - |-- Features_selection_plot.png..........................................................................................................4.5
     - |-- pca_plot.html ................................................................................................................................4.6
     - |-- pca_plot_log.html ..........................................................................................................................4.7
     - |-- pred_plot_Mid1_Hexb_Gm42418_LOG.html and png...................................................................4.8
     - |-- pred_plot_Mid1_Hexb_Gm42418.html and png.............................................................................4.9
     - |-- variance_plot.png...........................................................................................................................4.10

- |-- Submission............................................................................................................................................5
     - |-- 
     
- |-- data ......................................................................................................................................................6
     - |-- indexes.csv....................................................................................................................................6.1
     - |-- test.csv.gz.....................................................................................................................................6.2
     - |-- train.csv.gz....................................................................................................................................6.3

- |-- Google_Colab.ipynb ............................................................................................................................7
- |-- README.md ......................................................................................................................................8
- |-- Report.pdf............................................................................................................................................9

- |-- Visualisations.jl...................................................................................................................................10
- |-- clustermap.ipynb...............................................................................................................................11
- |-- cross_val.jl.........................................................................................................................................12
- |-- data_processing.jl..............................................................................................................................13
- |-- models.jl.............................................................................................................................................14

.....

**Contents :**

1. Best Models obtained

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


     4.1. Histogram showing the distributions of the experimental conditions in the training data.

     ![Histogram](https://user-images.githubusercontent.com/92453354/209339606-eecf75d2-13c0-4b3f-afce-ababb2add1aa.png)
     
     4.2. Supervised UMAP 2 components.
     
    ![UMAP_plot](https://user-images.githubusercontent.com/92453354/209356399-18eae0ff-56a0-45e8-94a8-fcd8592cb487.png)

     4.3. Heatmap of the 13 most important genes.
   
    ![clustermap](https://user-images.githubusercontent.com/92453354/209339284-b00ac23d-aa62-4537-801c-4c6c038fbfdc.png)

     4.4. Correlation plot between the four more important genes, according to a t-test. With a previous standardisation of the data.
   
     ![corr_plot_norm](https://user-images.githubusercontent.com/92453354/209339326-966fc15b-44e0-4af7-b798-ea211990f9da.png)

     4.5. Plot of the accuracy variation with different features selection techniques ( call-rates, mean variation and t-test).

     ![features](https://user-images.githubusercontent.com/92453354/209339267-4c4d14b3-060d-49a3-96fa-8863b1c268cd.png)
     
     4.6. PCA plot.
   
    ![PCA_plot](https://user-images.githubusercontent.com/92453354/209346393-9f74afb5-3b76-4e69-8e18-57e96dc98706.png)

     4.7. PCA plot with using the logarithm of the data.
     
    ![PCA_plot_log](https://user-images.githubusercontent.com/92453354/209346410-4cf7d86a-c4c4-4cb9-b67b-159b7ef76dd6.png)

     4.8. Spatial representation of Mid1, Hexb and Gm42418 genes.

     ![pred_plot_Mid1_Hexb_Gm42418](https://user-images.githubusercontent.com/92453354/209340284-783e260b-2990-48bf-9b19-2fba2bffdeb1.png)

     4.9. Spatial representation of Mid1, Hexb and Gm42418 genes with a previous logarithmic application.

     ![pred_plot_Mid1_Hexb_Gm42418_LOG](https://user-images.githubusercontent.com/92453354/209340295-43221ba3-ced3-4a42-8abb-073f17a52ec7.png)

     4.10. PCA explained variance plot

     ![variance_plot](https://user-images.githubusercontent.com/92453354/209339447-7af3d03b-a139-4c89-9a98-e3b635bb8952.png)



5. Best Kaggle Submissions.

6. Train, Test data and indexes of selected genes after cleaning.

7. Notebook julia to run a code on Google colab, need to download this file and save it in a google drive.

8. Readme.

9. Report that summarises your approach and findings

9. Code for all figures.

10. Python code for clustermap.

11. Template for feature selection tuning with a certain model.

12. Functions used for data processing and features engeneering.

13. Functions for evaluation and cross validation of machine learning techniques.

