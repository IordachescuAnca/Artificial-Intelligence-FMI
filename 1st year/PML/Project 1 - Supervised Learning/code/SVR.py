


import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import NuSVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error





#class used for manipulating the data text files
class Data:
    def __init__(self):
        self.train_data = None
        self.test_data = None
    
    #read the training data
    def add_train_data(self, path_train_data):
        df = open(path_train_data, 'r')
        
        train_data = []
        for line in df:
            line = line.strip().split('\t')
            #extracted the id, the whole sentence, the target word and the probability
            features = [line[0], line[1], line[4], line[9]]
            train_data.append(features)
        
        #create the train dataframe using the extracted columns from the train text file
        df_columns = ['id', 'sentence', 'target_word', 'output']
        self.train_data = pd.DataFrame(train_data, columns = df_columns)
    
    #read the test data
    def add_test_data(self, path_test_data):
        df = open(path_test_data, 'r')
        
        test_data = []
        for line in df:
            line = line.strip().split('\t')
            #extract the id, the whole sentence and the target word (there is no probability because it is the test set)
            features = [line[0], line[1], line[4]]
            test_data.append(features)
            
        #create the test dataframe using the extracted columns from the test text file    
        df_column = ['id', 'sentence', 'target_word']
        self.test_data = pd.DataFrame(test_data, columns = df_column)
    
    #get train data frame
    def get_train_data(self):
        return self.train_data
    
    #get test dataframe
    def get_test_data(self):
        return self.test_data




#class used for selecting the features
class FeatureEng:
    #corpus used for pronunciation and to get the syllabes of a word
    voc_syl = nltk.corpus.cmudict.dict()
    def __init__(self):
        self.features = []
    
    #this method is used to reduce the dimension of a large feature matrix
    #i used truncatedsvd because the params train_data and test_data are large sparse matrices
    @staticmethod
    def reduce_dimension(train_data, test_data, n):
        #declare the transformer by specifing the number of compunents
        t_svd = TruncatedSVD(n_components=n)
        #training and fit
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
                #choose only the first representation
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

    #main function for creating the features
    def create_features(self, df):
        features = []
        features_syl = []
        features_gram = []
        i = 0
        #the features are extracted only from the target word
        for target_word in df['target_word']:
            lower_target_word = target_word.lower()
            
            #get no of uppercase, vowels, consonants, number of words and characters
            no_upper = FeatureEng.number_upper(target_word)
            no_vowels = FeatureEng.number_vowels(target_word)
            no_cons = FeatureEng.number_consonant(target_word)
            no_words = FeatureEng.number_words(target_word)
            no_ch = len(target_word)
            
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
            features.append([no_upper, no_vowels, no_cons, no_words, no_ch, no_ch/no_words])
        
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
        
        #concatenate all the feature for test data
        X_test = np.concatenate((list_test_features[0], test_feature_syl), axis=1)
        X_test = np.concatenate((X_test, test_feature_gram), axis=1)
        
        return X_train, X_test
        #print(X_test.shape, X_train.shape)
        
    


#declare class data
d = Data()
#create dataframes for train and data sets
d.add_train_data('data/train_full.txt')
d.add_test_data('data/test.txt')



#get the dataframes
train_data = d.get_train_data()
test_data = d.get_test_data()


#create the 5folds for cross validation
kfold = KFold(n_splits = 5)
splits = kfold.split(train_data)


#declare class for text processing
fe = FeatureEng()
#different values for params C and nu
C = [0.01, 0.1, 1]
nu = [0.25, 0.35, 0.5]


fold = 0
#matrix with all scores
scores = []
for train_idx, val_idx in splits:
    #iterate through the splits
    fold += 1
    #create the train and val sets for cross validation
    X_train = train_data.iloc[train_idx]
    X_val = train_data.iloc[val_idx]
    #get the y_val probability of word being complex
    y_train = np.asarray(X_train['output']).astype(float)
    y_val = np.asarray(X_val['output']).astype(float)
    #get the features of train and val sets
    X_train, X_val = fe.get_features(X_train, X_val)
    
    score = []
    #iterate through different values of C
    for c_param in C:
        #iterate through fifferent values of nu
        for nu_param in nu:
            #declare the nusvr model
            svr = NuSVR(C=c_param, nu=nu_param)
            #train and predict
            svr.fit(X_train, y_train)
            predicted = svr.predict(X_val)
            #calculate the mean absolute error
            mae_score = mean_absolute_error(predicted, y_val)
            print("Fold {} - MAE for C={} and nu={} - {}".format(fold, c_param, nu_param, mae_score))
            score.append(mae_score)
    
    scores.append(score)



#print for table from documentation
scores = np.transpose(scores)
for line in scores:
    for score in line:
        print(np.round(score, 3), end=' ')
    print(np.round(np.mean(line), 3), end=' ')
    print()


#get feature for train and test data
X_train, X_test = fe.get_features(train_data, test_data)
#get the probability of word being complex
y_train = np.asarray(train_data['output']).astype(float)
#set the params for best performance
c_param = 1
nu_param = 0.5
#declare the model
svr = NuSVR(C=c_param, nu=nu_param)
#fit predict
svr.fit(X_train, y_train)
predicted = svr.predict(X_test)


#get the ids from the test set
id = np.asarray(train_data['id'])

#create submission with columns id and label
columns = ['id', 'label']
#create a list with two columns
output = list(zip(id, predicted))
#create dataframe
df_test = pd.DataFrame(output,columns = columns)
df_test.to_csv('SVR.csv')




