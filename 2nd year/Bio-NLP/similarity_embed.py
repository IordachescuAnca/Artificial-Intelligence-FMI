#!/usr/bin/env python
# coding: utf-8

# In[ ]:


embedder = SentenceTransformer('bert-base-nli-mean-tokens')


# In[ ]:


embedder = SentenceTransformer('stsb-bert-base')


# In[ ]:


embedder = SentenceTransformer('stsb-roberta-base')


# In[ ]:


embedder = SentenceTransformer('stsb-distilbert-base')


# In[ ]:


def similarity(posts, answer):
  cos_similarities = []
  for post in posts:
    corpus_embeds = embedder.encode(post)
    query_embeds = embedder.encode(answer)
    cos_similarity = util.pytorch_cos_sim(query_embeds, corpus_embeds)[0]
    cos_similarities.append(cos_similarity)
  return cos_similarities


# In[ ]:


with open('all_cosine_similarities.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['User','Post', 'Answer', 'Cosine Similarity'])


# In[ ]:


def select_answer(posts, answers, user):
  total = 0
  index = 0
  averages = []

  numDict = {0:'0', 1:'1a', 2: '1b', 3: '2a', 4: '2b', 5: '3a', 6: '3b'}
  
  for answer in answers:
      similarities = similarity(posts,answer)
      
      for i, similarity in enumerate(similarities):

        with open('all_cosine_similarities.csv', mode='a') as csv_file:
          csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          csv_writer.writerow([user, posts[i], answer, similarity.item()])
        csv_file.close()

        total = total + similarity.item()
        index += 1

      averages.append(total / index)
      total = 0
      index = 0
    
  if len(averages) > 4:
    return numDict.get(averages.index(max(averages)))

  return averages.index(max(averages))


# In[ ]:


def run_model(model):
  for user in users:
    posts = []
    index = 0
    filename = user + '-' + model + ".csv"
    
    with open(filename, mode='w') as csv_file:
      csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      csv_writer.writerow(['Question', 'Predicted Answer', 'Real Answer'])
      for a in answers:
        g = BDI_df[index].groupby(['User'])

        answer = g.get_group(user).head(1)['Class'].values[0]

        if posts == []:
          for row in g.get_group(user).iterrows():
            posts.append(row[1][3])

        index += 1
        csv_writer.writerow([("Q" + str(index)), select_answer(posts, a, user), answer])
    posts.clear()


# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


def convert_to_int(s):
  if type(s) == int:
    return s
  else:
    s = s[0]
    integer = int(s)
    return integer


# In[ ]:


run_model('bertNLI')


# In[ ]:


print("AHR: ", calc_AHR('bertNLI'))
print("ACR: ", calc_ACR('bertNLI'))
print("ADODL: ", calc_ADODL('bertNLI'))
print("DCHR: ", calc_DCHR('bertNLI'))


# ## **BERT STSb Model**

# In[ ]:



run_model('bert')


# In[ ]:


print("AHR: ", calc_AHR('stsb-bert'))
print("ACR: ", calc_ACR('stsb-bert'))
print("ADODL: ", calc_ADODL('stsb-bert'))
print("DCHR: ", calc_DCHR('stsb-bert'))


# ## **RoBERTa STSb Model**

# In[ ]:



run_model('roberta')


# In[ ]:


print("AHR: ", calc_AHR('roberta'))
print("ACR: ", calc_ACR('roberta'))
print("ADODL: ", calc_ADODL('roberta'))
print("DCHR: ", calc_DCHR('roberta'))


# ## **DistilBERT STSb Model**

# In[ ]:



run_model('distilbert')


# In[ ]:


print("AHR: ", calc_AHR('distilbert'))
print("ACR: ", calc_ACR('distilbert'))
print("ADODL: ", calc_ADODL('distilbert'))
print("DCHR: ", calc_DCHR('distilbert'))

