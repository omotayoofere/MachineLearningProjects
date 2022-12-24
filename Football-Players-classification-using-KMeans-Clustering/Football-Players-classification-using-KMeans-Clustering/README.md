## Football Players Classification using KMeans Clustering
The project utilizes K-Means clustering algorithm in identifying patterns for classifying football players based on predefined categories namely Age, Overall Value, Potential amd Wage rate.

## Installation

### Download the data
* The dataset used in this project was gotten on [kaggle](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmRrWE9DMmpCTEUyc1lDdjdncTM5aHMwd3ZDd3xBQ3Jtc0tsRHphcWxOTHhxMHJBbXBDMkVXd3lvRUxZc2hTdnNMWk5UUUFUeGhJRml3LTBLZWRrWUNMSjVZekMyZGFUR2FBTkh3eTdNNExfQUNzWjdEVFZqcnlsNzNmWW9JbmRMLWV4OTRNRlBFZkdqS2pLbE5VUQ&q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fstefanoleone992%2Ffifa-22-complete-player-dataset&v=lX-3nGHDhQg)

### Install the requirements
* Packages and modules used in this project can be gotten from the **requirements.txt** file by running
    ```
    pip install -r requirements.txt
    ```
## Project walk-through

* The first step of the project was selecting promising features that can be used to categorize football players. This step had use some domain knowledge to identify certain factors that can result in overall performance of a football player. Features - *overall, potential, wage_eur, value_eur, age* were selected while the remaining features were dropped.

* The entire dataset was then scaled to have a minimum value of 1 and max value of 10

* Initialize 5 random centroids based on selected features to work with with each point turned into a float data type

* Label data point based on how far each data point are from the centroids by looking at each point in the data frame and finding the Euclidean distance between that data point and each cluster center

* Update the centroids of each cluster by finding the geometric mean of the cluster centroid until the centroid are no longer getting updated.