# Splicing-Sites-Predicter
A gene  is a basic unit of heredity and a sequence of nucleotides in DNA or RNA that encodes the synthesis of a gene product, either RNA or protein.  Gene prediction or gene finding refers to the process of identifying the regions of genomic DNA that encode genes. This includes protein-coding genes as well as RNA genes, but may also include prediction of other functional elements such as regulatory regions. Gene finding is one of the first and most important steps in understanding the genome of a species once it has been sequenced.



Nucleotide distribution around splicing sites

![image.png](https://b3logfile.com/siyuan/1610205759005/assets/image-20210517215453-5avulne.png)

![image.png](https://b3logfile.com/siyuan/1610205759005/assets/image-20210517215449-94tsq3d.png)

## WAM

![Distribution_Probability.png](https://b3logfile.com/siyuan/1610205759005/assets/Distribution_Probability-20210602235755-1mc7fo9.png)

![Distribution_Probability_Cumsumed.png](https://b3logfile.com/siyuan/1610205759005/assets/Distribution_Probability_Cumsumed-20210602235747-irmi2lo.png)

![Plot_boxplot.png](https://b3logfile.com/siyuan/1610205759005/assets/Plot_boxplot-20210602235722-h05gaj2.png)

![ROC_plot.png](https://b3logfile.com/siyuan/1610205759005/assets/ROC_plot-20210603000901-9ckqyny.png)

![Plot_PR.png](https://b3logfile.com/siyuan/1610205759005/assets/Plot_PR-20210603000559-g71mn39.png)

## SVM

### Program Design

#### Step 1: feature extraction and data coding

Data processing method：

* Extraction of donor site length: in order to evaluate the impact of sample length on prediction, different lengths of singal series are selected to view different prediction results.
* Base encoding: adopts one-hot coding method, set up A code for,0,0,0 [1], G for,1,0,0 [0], C [0,0,1,0], T [0,0,0,1], Z (unknown base) for [0,0,0,0]
* Set negative sample: the negative sample is set to a sequence containing GT base pair but not donor site.
* Label：Set label to 1 for positive samples and 0 for negative samples.

Data extraction results:

* Each file in the training set and test set represents an eukaryotic gene. There is at least one exon and intron in each sequence, and all sites conform to the AG-GT rule, that is, all donor sites are GT, and all acceptor sites are AG. Finally, 2381 true donor signal  sequences are extracted from 462 training dataset.
* 2079 true donor signal sequences and 149255 pseudo signals are extracted from 570 files in the test dataset

#### Step 2: kernel function selection

The kernel functions of SVM are RBF, linear and poly. The training speed of linear is the fastest, and the effect is good in the case of linear separability. The training speed of RBF is the slowest, but it can fit better in the case of nonlinear, but it depends on parameters very much. The poly kernel is in the middle. This experiment will compare the performance of three kernel functions in the prediction of donor site

#### Step 3: Parameter learning

Superparameters  are parameters that cannot be directly learned by the model, which need to be set manually, or be found through hyperparameter optimization algorithms such as Bayesian optimization or Grid Search. For example, the superparameters of SVM include regularization parameter C, kernel coefficient containing gamma, degree, class Weight, cofe0, etc. When adjusting the superparameters, it might have been overfit on the test dataset, need to have a validation set so that training on the training set and evaluating on the validation set, and if it works  well, you can manage the final evaluation on the test dataset. Note that when adjusting parameters, we can't use test dataset for verification, because our purpose is to apply training model to unseen data.

I finally used  k-folded cross-validation  to evaluate the training effect of the model. In the k-folded cross-validation, the data we used are all the data in the training set .Randomly divide a dataset into k groups, or “folds”, of roughly equal size. Choose one of the folds to be the holdout set. Fit the model on the remaining k-1 folds. Calculate the validation scores on the observations in the fold that was held out. Repeat this process k times, using a different set each time as the holdout set. Calculate the overall validation scores to be the average of the scores.

GridSearHCV is used to find the best parameters and perform 3x cross-validation simultaneously

* C: it can adjust the extreme value of the penalty, control the size of the gap between the two lines, on behalf of the soft gap. When C is too large, the division is more strict, the generalization ability is weaker, and it is easy to over fit
* Gamma：

  * The larger the gamma value, the higher the dimension of the mapping and the more complex the model, which may make all points become support vectors
  * The smaller the gamma, the simpler the model。

#### Step 4: prediction and performance evaluation

Choosing the best parameters, I use three kernel functions to predict whether the sample contains donor site, and compare the performance of the three models and the difference between SVM and WAM by ROC and PR plot.

ROC curve is a curve reflecting the relationship between sensitivity and specificity. The closer the ROC curve is to the upper left corner, the higher the accuracy of the test. The point closest to the top left corner of the ROC curve is the best threshold with the least errors, and the total number of false positives and false negatives is the least. AUC is the area under the ROC curve. The meaning of AUC probability is to randomly take a pair of positive and negative samples, and the probability that the score of positive samples is greater than that of negative samples. AUC is robust to the unbalanced distribution of positive and negative samples, and it is a very common measure of classifier.

Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced(PR plot for short). In information retrieval, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned.When the distribution of positive and negative samples is unbalanced, the ROC curve remains unchanged, while the PR curve changes greatly. Compared with the ROC plot, the PR plotcan reflect the ability of the model to identify positive samples


### result
The results are surprisingly good.The AUC of Poly kernel is up to 0.9903, the RBF kernel and the linear kernel are up to 0.9896 and 0.9804 respectively. The PR-AUC of Poly kernel is up to  0.7007 , while the RBF kernel is  up to 0.6665.

![image.png](https://b3logfile.com/siyuan/1610205759005/assets/image-20210611172832-dxzvppn.png)

![](https://b3logfile.com/siyuan/1610205759005/assets/PR_plot-20210602065916-fs6co0b.png)


## BN

A Bayesian network, Bayes network, belief network, Bayes(ian) model or probabilistic directed acyclic graphical model is a probabilistic graphical model (a type of statistical model) that represents a set of random variables and their conditional dependencies via a directed acyclic graph (DAG). Bayesian networks are mostly used when we want to represent causal relationship between the random variables. Bayesian Networks are parameterized using Conditional Probability Distributions (CPD). Each node in the network is parameterized using P(node∣Pa(node)) where Pa(node) represents the parents of node in the network.

Choose window that contains 3 consecutive bases  upstream from the exon/intron boundary and 9 consecutive bases  downstream to the exon/intron boundary. Build a model by pgmpy. The network  obtained is shown in Figure 1.

![network.png](https://b3logfile.com/siyuan/1610205759005/assets/network-20210609212835-4n4fayy.png)

![ROC_plot.png](https://b3logfile.com/siyuan/1610205759005/assets/ROC_plot-20210612090114-8p3ozb6.png)

![PR_plot.png](https://b3logfile.com/siyuan/1610205759005/assets/PR_plot-20210612085857-0mzid4l.png)