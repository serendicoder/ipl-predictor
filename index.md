---
layout: default
---

# Introduction

Cricket is the most popular sport in the Indian subcontinent and the annual Indian Premier League (IPL) is one of its key events. Sports analysis remains a lucrative domain for professionals and hobbyists alike, evident in studies employing various ML methods across many sports tournaments with similar formats [[1]](https://docs.google.com/document/d/1zeap_CFpMrEivmf6gWQdVIxaBZ4qH-YPk99rw5QgvQ4/edit#heading=h.2tbvvj4y8gyy) [[2]](https://docs.google.com/document/d/1zeap_CFpMrEivmf6gWQdVIxaBZ4qH-YPk99rw5QgvQ4/edit#heading=h.yn323m7uyovb).

Barot et al [[3]](https://docs.google.com/document/d/1zeap_CFpMrEivmf6gWQdVIxaBZ4qH-YPk99rw5QgvQ4/edit#heading=h.4u6il7p52boz) analyze 4 years of IPL data to create a model that can predict the outcome of matches. The features extracted are team performance, location (home/away) and other external factors affecting a match. Another study [[1]](https://docs.google.com/document/d/1zeap_CFpMrEivmf6gWQdVIxaBZ4qH-YPk99rw5QgvQ4/edit#heading=h.2tbvvj4y8gyy) performs a binary classification to predict whether a team will make the NBA playoffs based on the performance in the regular season, using 25 years of match statistics. We aim to use two Kaggle IPL datasets [[4]](https://www.kaggle.com/datasets/manasgarg/ipl), [[5]](https://www.kaggle.com/datasets/vora1011/ipl-2008-to-2021-all-match-dataset), consisting of overall match statistics like the toss information, location, teams competing as well as ball-by-ball runs, wickets, batsman, bowler and commentary for similar prediction tasks.



# Problem Definition

There are two main goals of this project- <br/>
*  The primary objective of IPL teams is to qualify from the group stage to the playoffs which is also their indicator of success. Our project will predict the likelihood of IPL teams advancing to the playoffs. For this, we propose to train a supervised learning model with the most discriminative features extracted from the dataset.
*  Based on their past performances, each player in IPL is offered a contract from one of these categories- Grade-A+, Grade-A, Grade-B, Grade-C, and Non-Contracted. Our goal is to group the players in various contract categories using an unsupervised clustering algorithm. This analysis is useful for IPL team owners because a player's contract directly influences their bidding amount in player auctions.



# Methods

### Pre-Processing
1. Independent Features: We will calculate the covariance matrix to check if all features present in the dataset are independent. A low value of inter-feature correlation will indicate this.    
2. Noise Detection: We propose to use the DBSCAN algorithm to detect and remove possible outliers present in the dataset.
3. Dimensionality Reduction: We propose to use Principal Component Analysis to remove any unwanted features.
 
### Supervised Learning Task
1. 
2. 

### Unsupervised Learning Task
1. We can use K-means/GMM to cluster players based on performance to try to predict which category of contract they would be offered.



# Potential Results

We intend to accurately predict which team would advance in the IPL playoffs and perform quality clustering to group players in contract categories. To do so, we will evaluate the ML models proposed above and compare their performance based on the following evaluation metrics:
*  For the supervised learning task, we would evaluate the team performance predictions, in order to determine which model works the best for our data. We plan to score the models on the following metrics: Accuracy, Precision, Recall, AUC.    
*  For the unsupervised learning task, we plan to use Silhouette Coefficient and Davies-Bouldin Index to evaluate the performance of the clustering.



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

