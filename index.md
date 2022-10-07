---
layout: default
---

# Introduction

Cricket is the most popular sport in the Indian subcontinent and the annual Indian Premier League (IPL) is one of its key events. Sports analysis remains a lucrative domain for professionals and hobbyists alike, evident in studies employing various ML methods across many sports tournaments with similar formats [[1]](https://docs.google.com/document/d/1zeap_CFpMrEivmf6gWQdVIxaBZ4qH-YPk99rw5QgvQ4/edit#heading=h.2tbvvj4y8gyy) [[2]](https://docs.google.com/document/d/1zeap_CFpMrEivmf6gWQdVIxaBZ4qH-YPk99rw5QgvQ4/edit#heading=h.yn323m7uyovb).

Barot et al [[3]](https://docs.google.com/document/d/1zeap_CFpMrEivmf6gWQdVIxaBZ4qH-YPk99rw5QgvQ4/edit#heading=h.4u6il7p52boz) analyze 4 years of IPL data to create a model that can predict the outcome of matches. The features extracted are team performance, location (home/away) and other external factors affecting a match. Another study [[1]](https://docs.google.com/document/d/1zeap_CFpMrEivmf6gWQdVIxaBZ4qH-YPk99rw5QgvQ4/edit#heading=h.2tbvvj4y8gyy) performs a binary classification to predict whether a team will make the NBA playoffs based on the performance in the regular season, using 25 years of match statistics. We aim to use two Kaggle IPL datasets [[4]](https://www.kaggle.com/datasets/manasgarg/ipl), [[5]](https://www.kaggle.com/datasets/vora1011/ipl-2008-to-2021-all-match-dataset), consisting of overall match statistics like the toss information, location, teams competing as well as ball-by-ball runs, wickets, batsman, bowler and commentary for similar prediction tasks.



# Problem Definition

The goals of this project are two-fold:<br/>
* Firstly, we propose to **predict the likelihood of teams qualifying for the IPL playoffs**. We plan to train a supervised learning model, with the most discriminative features extracted from the dataset, to predict chances of each team advancing to the IPL playoffs. 
* Secondly, we propose to **group players in various contract categories based on their performance**. There are five possible player contract categories (Grade-A+, Grade-A, Grade-B, Grade-C, Non-Contracted), which directly influences the players bidding amount.  We propose to use an unsupervised clustering algorithm to tackle this problem.




# Methods

### Pre-Processing
1. **Independent Features:** We will calculate the covariance matrix to check if all features present in the dataset are independent. A low value of inter-feature correlation will indicate this. 
2. **Noise Detection:** We propose to use the DBSCAN algorithm to detect and remove possible outliers present in the dataset.
3. **Dimensionality Reduction:** We intend to use Principal Component Analysis to remove any unwanted features.
 
### Supervised Learning Task
To predict the likelihood of each team qualifying for the playoffs, we will use some of the popular linear classifiers such as Decision Trees, K-Neighbors classification, Naive-Bayes Classifier, etc.

### Unsupervised Learning Task
We plan to use K-means/GMM/DBScan to cluster players based on performance for predicting the contract category that is likely to be offered to them. 



# Potential Results

We intend to accurately predict which team would advance in the IPL playoffs and perform quality clustering to group players in contract categories. To achieve this, we will pre-process the data, perform a detailed qualitative and quantitative comparison of the proposed models, based on the following metrics:

* For the supervised learning task, we would evaluate the team performance predictions, in order to determine which model works the best for our data. We plan to score the models on the following metrics: Accuracy, Precision, Recall, ROC AUC.    
 
* For the unsupervised learning task, we plan to use Silhouette Coefficient and Davies-Bouldin Index to evaluate the performance of the clustering.




# References

1. Ma, Nigel. "NBA Playoff Prediction Using Several Machine Learning Methods." 2021 3rd International Conference on Machine Learning, Big Data and Business Intelligence (MLBDBI). IEEE, 2021.
2. Yaseen, Aliaa Saad, Ali Fadhil Marhoon, and Sarmad Asaad Saleem. "Multimodal Machine Learning for Major League Baseball Playoff Prediction." Informatica 46.6 (2022).
3. H. Barot, A. Kothari, P. Bide, B. Ahir and R. Kankaria, "Analysis and Prediction for the Indian Premier League," 2020 International Conference for Emerging Technology (INCET), 2020, pp. 1-7, doi: 10.1109/INCET49848.2020.9153972.
4. Garg, M (2016). Indian Premier League (Cricket), Version 5. Retrieved October 5, 2022 from [https://www.kaggle.com/datasets/manasgarg/ipl](https://www.kaggle.com/datasets/manasgarg/ipl) 
5. [https://www.kaggle.com/datasets/vora1011/ipl-2008-to-2021-all-match-dataset](https://www.kaggle.com/datasets/vora1011/ipl-2008-to-2021-all-match-dataset)
6. Vora, S (2022). IPL 2008 to 2022 All Match Dataset, Version 3. Retrieved October 5, 2022 from [https://www.kaggle.com/datasets/vora1011/ipl-2008-to-2021-all-match-dataset](https://www.kaggle.com/datasets/vora1011/ipl-2008-to-2021-all-match-dataset) 
7. A. Santra, A. Sinha, P. Saha and A. K. Das, "A Novel Regression based Technique for Batsman Evaluation in the Indian Premier League," 2020 IEEE 1st International Conference for Convergence in Engineering (ICCE), 2020, pp. 379-384, doi: 10.1109/ICCE50343.2020.9290569.



# Timeline




# Contribution

