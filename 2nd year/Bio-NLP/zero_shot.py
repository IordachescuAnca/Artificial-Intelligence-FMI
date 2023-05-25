import pandas as pd
from transformers import pipeline
import numpy as np

#answers from the questionnaire
file = open('/home/anca/Fac/bio_nlp/dataset/chestionar.txt',mode='r')
all_of_it = file.read()
file.close()
questions = all_of_it.split('\n\n')
all_answers = []
for question in questions:
    answers = question.split('\n')[1:]
    answers = [' '.join(x.split(' ')[1:]) for x in answers]
    all_answers.append(answers)


#ground truth
pth_eRisk_test_answers = '/home/anca/Fac/bio_nlp/dataset/test/Depression Questionnaires_anon.txt'
file = open(pth_eRisk_test_answers, "r")
test_answers = file.readlines()
test_answers = [x.strip() for x in test_answers]
test_answers = [x.split(' ') for x in test_answers]
print(len(test_answers))

users = [x[0] for x in test_answers]


classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


root_test = '/home/anca/Fac/bio_nlp/dataset/new_format_dataset_cleaned/test/'

for i in range(1, 22):
    file = root_test + 'question_' + str(i) + '.csv'
    df = pd.read_csv(file)
    candidate_labels = all_answers[i-1]

    for user in users:
        user_df = df.loc[df['subject'] == user]

        user_texts = user_df.text.values


        labels = []
        for text in user_texts:
            sequence_to_classify = text
            print(sequence_to_classify)
            print(candidate_labels)
            probs = classifier(sequence_to_classify, candidate_labels)['scores']
            print(probs)
            print(np.argmax(probs))
            labels.append(np.argmax(probs))
        

        hist = np.bincount(labels)
        maj_vote = np.argmax(hist)
        