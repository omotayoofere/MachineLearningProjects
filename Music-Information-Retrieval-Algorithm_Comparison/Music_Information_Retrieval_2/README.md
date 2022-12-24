# Music Information Retrieval (Contd.)
In continuation of the approach utilized in [**Music Information Retrieval**](https://github.com/omotayoofere/Muisc_Information_Retrieval) where the CNN algorithm was used in classifying the [**spectogram**](https://vibrationresearch.com/blog/what-is-a-spectrogram/) of the audio files.The accuracy gotten in this approach oscillated in the 40% range, this questons the suitability of the CNN model on this dataset as even tunning the CNN hyper parameters and performing a cross-validation did not improve the accuracy fo the model.

To acheive a better accuracy score on this dataset, it is necessary to try a handful of other algorithms and see the best performing model on this dataset. In this project, classification algorithms namely; Random Forest, Logistic Regression, Support Vector Machines (SVM), K-Nearest Neigbours and Artificial Neural Network (ANN) were utilized in an [**sklearn pipeline**](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). 

The motivation behind this study is to achieve a better score for the accuracy metric in the classification of music data by exploring handful for machine learning models. For a given song, the music classifier predicts its genre based on relevant musical attributes


### Problem Definition
* The project definition remains exploring multi-class classification, that is, categorizing each music sample into either of the ten (10) labels available. 
### What is our data
* I have choosen the famous GTZAN dataset which is available on [kaggle](/https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
### Libraries and Dependencies
* I will be utilizing google collab on this project and mounting the dataset on Google drive. Also, I have listed all necessary libraries and dependencies needed for this project.
``` python
  #For manipulating data
  import numpy as np
  import pandas as pd

  #For Preprocessing
  from sklearn.preprocessing import LabelEncoder, StandardScaler
  from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
  from sklearn import metrics

  from tensorflow.keras import layers, models
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Activation, Dropout

  #For modelling
  from tensorflow.keras import layers
  from tensorflow.keras import models
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.callbacks import EarlyStopping
  from sklearn.pipeline import Pipeline
  from sklearn.svm import SVC
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.decomposition import PCA

  #For tuning HyperParameters
  from sklearn.model_selection import GridSearchCV
```
[**Librosa**](https://pypi.org/project/librosa/), a Python module to analyze audio signals and music signal processing, provides implementations of a variety of common functions used throughout the field of music information retrieval. It allows extraction of certain features that are important and useful to retrieving information from music. 

[**ARTICLE**](https://librosa.org/) shows how this module can be installed and imported into the environment.

``` python
  #importing librosa
  import librosa
```
#### Extracting features from audio files
The audio folder consisting of the 10 genres with each genre having a 100 copies of **.wav** audio files was downloaded from kaggle. 

Iterating through each audio files in each sub-folders, reading the files using the installed librosa package and extracting information deemed as a feature before writing into a **.csv** file

``` python
  def load_dataset(src_folder='C:/Users/google cloud/Desktop/genres', dest_filename='data.csv'):
      #opens a new csv file named data.csv
      file = open('data.csv', 'w', newline='')

      with file:#working with created file

          writer = csv.writer(file)#writing into data.csv file

          writer.writerow(header)#making headers we already declared the title of columns in data.csv
```

  ``` python
      #creating like a list of genre type and spliting them based on spaces that exists between them
      genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
      #for each of the splitted entity
      for g in genres:
          #pointing to the names of file in a particular directory
          for filename in os.listdir(src_folder + '/' + g):
              #getting a particular file under each genre
              songname = src_folder + '/' + g + '/' + filename
              #loading the particular file using librosa, with duration set to 30secs
              y, sr = librosa.load(songname, mono=True, duration=30)
              #getting chroma stft of a particular file
              chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
              #getting spectral centroid of a particular file
              spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
              #getting spectral bandwidth of a particular file
              spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
              #getting spectral roll-off of a particular file
              rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
              #getting zero crossing rate of a particular file
              zcr = librosa.feature.zero_crossing_rate(y)
              #getting rmse of a particular file
              rmse = librosa.feature.rms(y=y)[0]
              #getting 20-mfcc(s) of a particular file
              mfcc = librosa.feature.mfcc(y=y, sr=sr)
              #appending all the mean of the calculated features for a particular file in a row of data.csv file except the mfcc(s) at this point
              to_append = filename + ' ' + np.mean(chroma_stft) + ' ' + np.mean(rmse) + ' ' + np.mean(spec_cent) + \
              ' ' + np.mean(spec_bw) + ' ' + np.mean(rolloff) + ' ' + np.mean(zcr)

              #getting the mean of each of the 20 mfcc(s) of a particular file
              for e in mfcc:
                  #appending this set of mfcc mean to the already existing data.csv
                  to_append += ' ' + np.mean(e)
              #getting the label of each of the file and attaching it to its respective row
              to_append += ' ' + g
              #opening the created data.csv file and saving all our stuff into it
              file = open(dest_filename, 'a', newline='')
              with file:
                  writer = csv.writer(file)
                  writer.writerow(to_append.split())
```

#### Loading data
After extracting these features and writing to a **.csv**, the resulting **.csv** file is saved and stored on google drive for remote and easy availability.
Loading the data to google colab from google drive, we import the supported module and read the csv file as shown below.

``` python
  from google.colab import drive
  drive.mount('/content/gdrive')
  
  data_df = pd.read_csv('/content/gdrive/MyDrive/Datasets/features_30_sec.csv')
```

#### Preprocessing data
The **filename** and **length** columns contains as much distinct entries and an all-uniform entries respectively.
This explains that these two columns do not promise much value in the training of this model and thus would be dropped

``` python
  data_df = data_df.drop(['filename','length'], axis=1)
```
The target columns contains categorical variables that needs to be encoded into numerical variables for our model to understand

``` python
  label_cols = data_df.iloc[:,-1]
```
 
 ![alt text](https://github.com/omotayoofere/Music_Information_Retrieval_2/blob/main/unique.png "Label Categories")
 
Next, we encode these labels into numerical variables such that every category gets a numerical value assigned to it
``` python
  #getting unique items in output list
  unique_labels = np.unique(label_cols)

  encoder = LabelEncoder()

  cols_encoding = encoder.fit_transform(label_cols)
  list_encoding = encoder.fit_transform(unique_labels)
```
Splitting the dataset into dependent and independent variables

``` python
  X = data_df.iloc[:,:-1]
  y = cols_encoding
```
Creating the pipeline with a 
  - StandardScaler
  - Principal Component Analysis
  - 

``` python
pipeline=Pipeline([
    ('scaler', StandardScaler()),
    ("PCA", PCA(n_components=0.99)),
    ('clf', LogisticRegression())
])
pipeline.steps
```
