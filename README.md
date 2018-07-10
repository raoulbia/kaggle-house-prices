### Practicing Machine Learning: Linear Regression

#### Introduction

The goal of this repository is to implement **linear regression** without using 
off-the-shelf machine learning libraries. My DIY implementation of logistic regression is based 
on the 
[Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) 
course by Stanford University. The dataset was sourced from the 
[Kaggle House Price prediction](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition

**A note on Feature Engineering**

In order to focus on the machine learning aspect, I chose not not to spend time
on the (very important) step of feature engineering. Instead I use publicly available code to 
pre-process the raw data before submitting it to my algorithms. 
The source for any data wrangling code I use is available in the *data_clean_name.py* files.
 
<br>


#### Implementation Notes

I immplemented linear regression using *regularized gradient descent*. 

To evaluate the algorithm I implemented:
 
 * a *cost history curve* function to check whether cost converges to a minimum 
 * a *learning curve* to analyze the algorithm in terms of *bias* and *variance*

<br>

#### House Price predictions scored by Kaggle

* hyper-parameters: 
  * num iterations = 1000
  * learning rate = 0.3 
  * regularization term = 1.2
  
* Results: 
  * root mean squared logarithmic error 0.35
  * leaderboard position  4674 (as of 07/2018)

<br>

#### Notes 

* Feature normalization **is required** for linear regression.

* Regularization is applied to the computation of gradients **and** to the computation of the 
cost.

<br>

#### Further Reading

* [example of another DIY implementation of linear regression](https://github.com/itsrandeep/linear_regression)

* Linear Regression
  * [Linear regression models](https://people.duke.edu/~rnau/411regou.htm)
  * [Real Estate Price Prediction with Regression and Classification](http://cs229.stanford.edu/proj2016/report/WuYu_HousingPrice_report.pdf)
  * [Linear regression blogpost]
  * [House price prediction blogpost](https://towardsdatascience.com/regression-analysis-model-used-in-machine-learning-318f7656108a)
  * [Kaggle datasets for regression analysis](https://www.kaggle.com/rtatman/datasets-for-regression-analysis)
  
* Learning Rate
  * [High vs. low learning rate](https://stackoverflow.com/questions/34828329/pybrain-overflow-encountered-in-square-invalid-value-encountered-in-multiply)  

* Data Wrangling
  * [Notebook with data analysis examples](https://github.com/Shreyas3108/house-price-prediction)
