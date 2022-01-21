

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

class Data:
  #class used for manipulating data
  def __init__(self, file):
    #read the data and put it in a dataframe
    self.df = pd.read_csv(file, encoding="latin_1")
    self.get_data()
  
  def get_data(self):
    #split the x and y from the dataframe
    self.y = []
    self.X = []
    for _, row in self.df.iterrows():
      #get the message text
      message = row['Message']
      label = row['Category']
      
      if message not in self.X:
        self.X.append(message)
        #convert the labels into categorical
        if label == 'ham':
          self.y.append(0)
        else:
          self.y.append(1)

    self.X = np.asarray(self.X)
    self.y = np.asarray(self.y)

class FeatureEng:
  #class used for feature engineering
  def __init__(self):
    #declare the set of english stop words
    self.stop_words = set(stopwords.words('english'))
    #declare the lemmatizer
    self.lemmatizer = WordNetLemmatizer()
  
  def preprocess_single_message(self, message):
    #lower the message
    message = message.lower()
    #get the characters that are alpha or space
    prep_message = ''.join(char for char in message if char.isalpha() or char == " ")
    #split the message into words
    words = word_tokenize(prep_message)

    preprocess_words = []
    for w in words:
      #remove the stop words from the message
      if w not in self.stop_words:
        preprocess_words.append(w)
    
    preprocess_text = ""
    #concatenate all the remaining lemmatized words
    for p_w in preprocess_words:
      preprocess_text = preprocess_text + self.lemmatizer.lemmatize(p_w) + " " 

    return preprocess_text

  
  def preprocess_data(self, X):
    #method used to preprocess a list of messages
    preprocess_lines = []
    for line in X:
      #apply preprocessing for each message
      preprocess_line = self.preprocess_single_message(line)
      preprocess_lines.append(preprocess_line)
    return preprocess_lines

#declare data
data = Data('/content/gdrive/MyDrive/IA/PML/dataset.csv')

#split the data: 90% training and 10% testing
split = 0.1
X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=split)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#declare feature engineering class and preprocess the messages from both the training and test set
fe = FeatureEng()
X_train = fe.preprocess_data(X_train)
X_test = fe.preprocess_data(X_test)

#declare the bag of words rith unigrams and bigrams and apply on the training and test set
vectorizer = CountVectorizer(max_features = 100, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

print(X_train.shape, X_test.shape)

#declare 3 arrays that will contain the clusters, the value of epsilon and the value of silhouettes score
clusters = []
epsilon = []
silhouettes = []
#rez is the best configuration for dbscan
rez = [0, 0, 0]
for eps in np.linspace(0.5, 1.5, 10):
    #fit the dbscan
    dbscan = DBSCAN(eps = eps).fit(X_train)
    #get the labels
    labels = dbscan.labels_
    #append the number of clusters
    clusters.append(np.max(labels)+1)
    #append the epsilin value
    epsilon.append(eps)
    #calcule the silhouette score and append
    silhouette = silhouette_score(X_train, dbscan.labels_)
    #save the best dbscan model with the greatest value of silhouette score
    if silhouette > rez[0]:
      rez[0] = silhouette
      rez[1] = eps
      rez[2] = np.max(labels)+1
    silhouettes.append(silhouette)

#plot the epsilon and silhouettes score
#it is obvious that the best value is 1.5
plt.plot(epsilon, silhouettes, 'x-')
plt.xlabel('Epsilon')
plt.ylabel('Sillhouette score')
plt.title('Sillhouette Method')
plt.show()

print("The best configuration for DBSCAN: Eps: {} with cluster: {}, sillhouette score: {} ".format(rez[1], rez[2], rez[0]))

class DBScan:
  #implement a dbscan models that works even if the number of clusters is different than the number of actual classes
  def __init__(self, eps = 2.5):
    #declaration of the model
    self.eps = eps
    self.dbscan = DBSCAN(eps = self.eps, min_samples = 5) 
  
  def fit(self, data):
    #fit the model to the data
    self.dbscan.fit(data)
    #get the number of clusters
    self.n_clusters = np.max(self.dbscan.labels_)+1
    #obtain the predicted labels of the cluster
    self.pred_label_cluster = dbscan.labels_
    self.adapt_labels(data)

  
  def convert_using_cm(self, y_train, n_true_labels = 2):
    #create the confusion matrix with the size equal to number of clusters x number of actual classes
    cm = np.zeros((n_true_labels, self.n_clusters))
    for i in range(len(y_train)):
      y_train_label = y_train[i]
      y_pred = self.pred_label_cluster[i]
      #adding +1 to m[i][j] where i represents the class and j the number of cluster
      cm[y_train_label][y_pred] += 1
    #create transpose
    cm = np.transpose(cm)

    convert = []
    #get the index of maximum value per rows
    for i in range(self.n_clusters):
      convert.append(np.argmax(cm[i, :]))
    
    return convert
  
  def adapt_labels(self, data, n_true_labels = 2):
    #for each cluster assign the correspondent data
    self.n_true_labels = n_true_labels
    self.clusters = dict()
    for cluster in range(self.n_clusters):
      self.clusters[cluster] = []
      for i in range(len(self.pred_label_cluster)):
        label = self.pred_label_cluster[i]
        #if the label is equal to the actual cluster, add to the cluster
        if cluster == label:
          self.clusters[cluster].append(data[i])


  def print_conversion(self, labels):
    #print conversion of each cluster to the actual class
    convert = self.convert_using_cm(labels, self.n_true_labels)
    for i in range(len(convert)):
      print("{}: {}".format(i, convert[i]))
  
  def get_modified_labels(self, new_data, labels):
    pred_cluster_label = []
    for features in new_data:
      sol_dist = []
      for cluster_n in range(self.n_clusters):
        #compute the minimum distance between the test features samples and cluster samples
        diff = self.clusters[cluster_n] - features
        dist = diff * diff
        dist = np.sum(dist, axis=1)
        sol_dist.append(np.min(dist))
      pred_cluster_label.append(np.argmin(sol_dist))
    
    #confusion matrix conversion
    convert = self.convert_using_cm(labels, self.n_true_labels)
    predicted_labels = []
    for i in range(len(pred_cluster_label)):
      #for each predicted label of test data the converted class is assigned
      label = pred_cluster_label[i]
      predicted_label = convert[label]
      predicted_labels.append(predicted_label)

    return predicted_labels

#declare the dbscan model and fit
my_dbscan = DBScan(eps=1.5)
my_dbscan.fit(X_train)

#get the predicted labels for the test data
predicted = my_dbscan.get_modified_labels(X_test, y_train)

#print the converted clusters to actual classes
my_dbscan.print_conversion(y_train)

print("Acc for the Kmeans model: {}".format(accuracy_score(y_test, predicted)))

# print the confusion matrix for dbscan model
confm = confusion_matrix(y_test, predicted)

figure, ax = plot_confusion_matrix(conf_mat = confm,
                                   colorbar = False)
plt.show()

#declare the supervised learning SVM model
svm = SVC()
#train and get the predicted labels
svm.fit(X_train, y_train)
svm_predicted_labels = svm.predict(X_test)

print("Acc for the SVM model: {}".format(accuracy_score(y_test, svm_predicted_labels)))

# print the confusion matrix for kmneans model
confm = confusion_matrix(y_test, svm_predicted_labels)

figure, ax = plot_confusion_matrix(conf_mat = confm,
                                   colorbar = False)
plt.show()