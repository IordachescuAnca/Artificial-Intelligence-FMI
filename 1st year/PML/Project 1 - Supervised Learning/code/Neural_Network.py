

import numpy as np
import pandas as pd
import nltk
import gensim.downloader as api
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


#class used for manipulating the data text files
class Data:
    #the corpus used for training the w2v is text8
    name_corp = "text8"
    def __init__(self):
        self.train_data = None
        self.test_data = None
        #get the sentences of the corpus
        self.corp = list(api.load(Data.name_corp))
    
    #read the training data
    def add_train_data(self, path_train_data):
        df = open(path_train_data, 'r')
        
        train_data = []
        for line in df:
            line = line.strip().split('\t')
            #extracted the id, the whole sentence, the begin and end index of target word in sentence, the target word and the probability
            features = [line[0], line[1], line[2], line[3], line[4], line[9]]
            train_data.append(features)
        
        #create the train dataframe using the extracted columns from the train text file
        df_columns = ['id', 'sentence', 'begin', 'end', 'target_word', 'output']
        self.train_data = pd.DataFrame(train_data, columns = df_columns)
    
    #read the test data
    def add_test_data(self, path_test_data):
        df = open(path_test_data, 'r')
        
        test_data = []
        for line in df:
            line = line.strip().split('\t')
            #extracted the id, the whole sentence, the begin and end index of target word in sentence and the target word 
            features = [line[0], line[1], line[2], line[3], line[4]]
            test_data.append(features)
            
        #create the test dataframe using the extracted columns from the train text file
        df_column = ['id', 'sentence', 'begin', 'end', 'target_word']
        self.test_data = pd.DataFrame(test_data, columns = df_column)
    
    #get train data frame
    def get_train_data(self):
        return self.train_data
    
    #get test data frame
    def get_test_data(self):
        return self.test_data
    
    #get the words from the sentences of the training data in order to add them to train the w2v
    def create_words(self):
        sentences = []
        #iterate through all sentences
        for sentence in self.train_data['sentence']:
            #if the sentence was not already chosen
            if sentence.lower() not in sentences:
                sentences.append(sentence.lower())
                
        voc = []
        #iterate through new sentences
        for sentence in sentences:
            #get the words
            words = nltk.word_tokenize(sentence)
            #concatenate them to voc
            voc.append(words)
        self.voc = voc


#class used for selecting the features
class FeatureEng:
    #corpus used for pronunciation and to get the syllabes of a word
    voc_syl = nltk.corpus.cmudict.dict()
    #declare w2v as static attribute
    w2v = None
    def __init__(self):
        self.features = []
    
    #train w2v using the corp and the voc created in class data
    #the size of word embeddings = 100
    @staticmethod
    def train_w2v(data):
        voc = data.voc
        corp = data.corp
        FeatureEng.w2v = Word2Vec(corp + voc, vector_size=100)
    
    #this method is used to reduce the dimension of a large feature matrix
    #i used truncatedsvd because the params train_data and test_data are large sparse matrices
    @staticmethod
    def reduce_dimension(train_data, test_data, n):
        t_svd = TruncatedSVD(n_components=n)
        train_data = t_svd.fit_transform(train_data)
        test_data = t_svd.transform(test_data)
        return train_data, test_data
    
    #method that calculates the number of uppercase or lowercase vowels in a string
    @staticmethod
    def number_vowels(string):
        cnt = 0
        vowels = 'aeiou'
        for x in string:
            if x.lower() in vowels:
                cnt += 1
        return cnt
    
    #method that calculates the number of uppercase or lowercase consonants in a string
    @staticmethod
    def number_consonant(string):
        cnt = 0
        vowels = 'aeiou'
        for x in string:
            if x.isalpha():
                if not x.lower() in vowels:
                    cnt += 1
        return cnt
    
    #method that calcules the number of words using tokenize from nltk
    @staticmethod
    def number_words(string):
        words = nltk.word_tokenize(string)
        return len(words)
    
    #method that gets all the syllabes that appears in a string 
    @staticmethod
    def get_syl(string):
        syls = []
        words = nltk.word_tokenize(string)
        for word in words:
            if word in FeatureEng.voc_syl:
                syls = syls + FeatureEng.voc_syl[word][0]
        
        concatenated_syls = ""
        for i in range(len(syls)):
            concatenated_syls = concatenated_syls + syls[i] + " "
        
        return concatenated_syls
    
    #method that calcules the number of uppercase letters
    @staticmethod
    def number_upper(string):
        cnt = 0
        for x in string:
            if 'A' <= x <= 'Z':
                cnt += 1
        return cnt
    
    #method that gets all the ngrams (with n specified) of a string
    @staticmethod
    def get_ngram(string, n):
        n_grams = []
        for i in range(len(string) - n + 1):
            gram = ""
            for j in range(n):
                gram = gram + string[i+j]
            
            n_grams.append(gram)
        
        return n_grams
    
    #method that gets a sparse matrix that tells how relevant words are in documents
    @staticmethod
    def tf_idf(train_data, test_data):
        vec = TfidfVectorizer()
        train_data = vec.fit_transform(train_data)
        test_data = vec.transform(test_data)
        return train_data, test_data
    
    #method that calcules the sum of word embeddings 
    @staticmethod
    def word_embeddings_target_word(string):
        words = nltk.word_tokenize(string)
        avg = np.zeros(100)
        
        for word in words:
            #add onlu the words that appears in the w2v vocabulary
            if word in FeatureEng.w2v.wv.key_to_index:
                avg = avg + FeatureEng.w2v.wv[word]
                
        #divide them by number of words
        avg = avg / len(words)
        #return list of the word embeddings
        return avg.tolist()

    #main function for creating the features
    def create_features(self, df):
        features = []
        features_syl = []
        features_gram = []
        
        #the features are extracted only from the target word and sentence
        for index, row in df.iterrows():
            #get begin and end indexes for the target word
            begin = int(row['begin'])
            end = int(row['end'])
            #get sentence
            sentence = row['sentence']
            #get content word
            target_word = row['target_word']
            #create the context for the target word 
            context = sentence[0:begin] + sentence[end:]
            
            lower_target_word = target_word.lower()
            
            #get no of uppercase, vowels, consonants, number of words and characters
            no_upper = FeatureEng.number_upper(target_word)
            no_vowels = FeatureEng.number_vowels(target_word)
            no_cons = FeatureEng.number_consonant(target_word)
            no_words = FeatureEng.number_words(target_word)
            no_ch = len(target_word)
            
            #get the word embeddings for the target word
            we_target_word = FeatureEng.word_embeddings_target_word(target_word)
            #get the word embeddings for the context
            we_context = FeatureEng.word_embeddings_target_word(context)
            
            #gets the syllabes
            syls = FeatureEng.get_syl(lower_target_word)
            features_syl.append(syls)
            
            #get the 2gram
            gram2 = FeatureEng.get_ngram(target_word, 2)
            #get the 3gram
            gram3 = FeatureEng.get_ngram(target_word, 3)
            #get the 4gram
            gram4 = FeatureEng.get_ngram(target_word, 4)
            
            #concatenate all the ngrams
            grams = gram2 + gram3 + gram4
            gram_conc = ""
            for i in range(len(grams)):
                gram_conc = gram_conc + grams[i] + ' '
            features_gram.append(gram_conc)
            
            #create lists of features
            features.append([no_upper, no_words, no_ch, no_ch/no_words] + we_target_word + we_context)
        
        return [np.asarray(features), np.asarray(features_syl), np.asarray(features_gram)]
    
    def get_features(self, train_df, test_df):
        #create features for both train and test data
        list_train_features = self.create_features(train_df)
        list_test_features = self.create_features(test_df)
        
        #get tf idf features from syllabes and reduce dimension of matrix
        train_feature_syl = list_train_features[1]
        test_feature_syl = list_test_features[1]
        train_feature_syl, test_feature_syl = FeatureEng.tf_idf(train_feature_syl, test_feature_syl)
        train_feature_syl, test_feature_syl = FeatureEng.reduce_dimension(train_feature_syl, test_feature_syl, 15)
        #print(train_feature_syl.shape, test_feature_syl.shape)
        
        #get tf idf featyres from grams and reduce dimension of matrix
        train_feature_gram = list_train_features[2]
        test_feature_gram = list_test_features[2]
        train_feature_gram, test_feature_gram = FeatureEng.tf_idf(train_feature_gram, test_feature_gram)
        train_feature_gram, test_feature_gram = FeatureEng.reduce_dimension(train_feature_gram, test_feature_gram, 100)
        #print(train_feature_gram.shape, test_feature_gram.shape)
        
        #concatenate all the features for train data
        X_train = np.concatenate((list_train_features[0], train_feature_syl), axis=1)
        X_train = np.concatenate((X_train, train_feature_gram), axis=1)
        
        #concatenate all the features for test data
        X_test = np.concatenate((list_test_features[0], test_feature_syl), axis=1)
        X_test = np.concatenate((X_test, test_feature_gram), axis=1)
        
        return X_train, X_test
        #print(X_test.shape, X_train.shape)
        
    


#function created for training a neural network
def training_model(X_train, y_train, X_val, y_val, patience=10, opt = 'Adam', model='m3',min_lr=0.000001, lr=0.0001, factor = 0.01, epoch=100, batch=32, reduceLR=True):
    #initialize the reduceLR
    rlr = keras.callbacks.ReduceLROnPlateau(factor=factor, patience=patience, min_lr=min_lr)
    
    #Adam optimizer declared 
    if opt == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    #architecture 1
    if model == 'm3':
        model = keras.Sequential([
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(200, activation='relu'),
        layers.Dense(1)
    ])
    #arhitecture 2
    elif model == 'm2':
        model = keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),    
        layers.Dense(1)
    ])
    #arhitecture 3
    elif model == 'm1':
        model = keras.Sequential([
        layers.Dense(200, activation='relu'),
        layers.Dense(1)
    ])
    
    #compline de model using MAE loss and the Adam optimizer
    model.compile(loss=keras.losses.MAE, optimizer=optimizer)
    
    #use the reduceLR if needed
    if reduceLR:
        history = model.fit(X_train, y_train, batch_size=batch, epochs=epoch, validation_data=(X_val, y_val), callbacks=[rlr],)
    else:
        history = model.fit(X_train, y_train, batch_size=batch, epochs=epoch, validation_data=(X_val, y_val),)
    
    #return the history, predicted prob and the model
    return history, model.predict(X_val), model
    



#declare class data
d = Data()
#create dataframes for train and data sets
d.add_train_data('data/train_full.txt')
d.add_test_data('data/test.txt')
#create also the voc for training data
d.create_words()



#get the dataframes
train_data = d.get_train_data()
test_data = d.get_test_data()



#declare class for text processing
fe = FeatureEng()
#train the word2vec model
fe.train_w2v(d)



#create the 5folds for cross validation
kfold = KFold(n_splits = 5)
splits = kfold.split(train_data)

#scores for the cross validation
arr_scores = []

#iterate through the splits
for train_idx, val_idx in splits:
    #create the train and val sets for cross validation
    X_train = train_data.iloc[train_idx]
    X_val = train_data.iloc[val_idx]
    #get the y_val probability of word being complex
    y_train = np.asarray(X_train['output']).astype(float)
    y_val = np.asarray(X_val['output']).astype(float)
    #get the features of train and val sets
    X_train, X_val = fe.get_features(X_train, X_val)
    
    #declare the model specifing patience, learning rade, if use reducelr, model of arhitecture, no of epoch
    #returns predicted probabilitues
    _, predicted, _ = training_model(X_train, y_train, X_val, y_val,patience=5,lr=0.00001, reduceLR=True, epoch=100, model='m3')
    #compute mean absolute score
    mae = mean_absolute_error(predicted, y_val)
    #add the score to list of scores
    arr_scores.append(mae)
    
    



#print the 5fold loss
print(np.round(arr_scores, 3))


#print mean for cross validation
print(np.round(np.mean(np.round(arr_scores, 3)),3))


#get the true values of probabilities
y = np.asarray(train_data['output']).astype(float)
X = train_data
split = 0.1
#split the set into train data and validation data 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split)

#get features of train and validation data
X_train, X_val = fe.get_features(X_train, X_val)

#train the model using the params for best performance
history, predicted, model = training_model(X_train, y_train, X_val, y_val,patience=5,lr=0.0001, reduceLR=True, epoch=100, model='m3')


#plot the loss and val loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#set labels
plt.ylabel('MAE loss')
plt.xlabel('number of epoch')
#set legend
plt.legend(['train_loss', 'validation_loss'])
plt.show()
#seems that our model overfits


#get the features for test data
_, X_test = fe.get_features(train_data, test_data)


#predict the probabilities for test set
predicted = model.predict(X_test)
predicted = predicted.reshape(-1)
new_pred = []
#the probabilities should be in interval [0,1]
for x in predicted:
    if x < 0:
        new_pred.append(0)
    elif x > 1:
        new_pred.append(1)
    else:
        new_pred.append(x)

new_pred = np.asarray(new_pred)


#get the ids from the test set
id = np.asarray(train_data['id'])


#create submission with columns id and label
columns = ['id', 'label']
#create a list with two columns
output = list(zip(id, new_pred))
#create dataframe
df_test = pd.DataFrame(output,columns = columns)
df_test.to_csv('NN.csv')



