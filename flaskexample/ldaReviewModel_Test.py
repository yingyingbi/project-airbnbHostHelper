
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim


# In[ ]:

def ldaReviewModel_Test(user_review_texts):
    userReview = user_review_texts
    
    ## parameter
    Num_Words = 15
    Num_Topics = 20
    selected_topic_id = [2,8,12,13,14,16,17,18]
    num_selected_topics = len(selected_topic_id)
    selected_topic_title = {2:'Tourist Atrraction', 8:'Interior',12:'Accuracy',13:'Neighbourhood',14:'Experience',16:'Communication',17:'View',18:'Transportation'}
    min_prob = 0.00001
    
    ## Clean raw reviews
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = get_stop_words('en')
    ## high frequency words
    custom_stop = ['stay','great','place','us','love','host','recommend','everything','francisco','perfect','locat','apart', 'nice','good','definit','realli','room','didn']
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    
    # list for tokenized documents in loop
    user_texts = []
    # loop through document list
    for i in userReview:
    
        # clean and tokenize document string
        user_raw = str(i).lower()
        user_tokens = tokenizer.tokenize(user_raw)

        # remove stop words from tokens
        user_stopped_tokens = [i for i in user_tokens if not i in en_stop and not i in custom_stop]
    
        # stem tokens
        user_stemmed_tokens = [p_stemmer.stem(i) for i in user_stopped_tokens]
        user_filtered_tokens = [i for i in user_stemmed_tokens if len(i) > 3]
        # add tokens to list
        user_texts.extend(user_filtered_tokens)
    
    ## Load LDA
    # turn our tokenized documents into a id <-> term dictionary
    user_dictionary = corpora.Dictionary([user_texts])
    # convert tokenized documents into a document-term matrix
    user_corpus = user_dictionary.doc2bow(user_texts)
    lda_trained = models.LdaModel.load('lda20.model')
    topics = lda_trained.show_topics(num_topics=Num_Topics, num_words=Num_Words, formatted=False)
    topics_table = {}
    for ti, topic in enumerate(topics):
        topics_table[ti] = ', '.join([v[0] for v in lda_trained.show_topic(ti, Num_Words)])
    
    ## Apply LDA on user reviews
    selected_topic_prob = {}
    for ID in selected_topic_id:
        selected_topic_prob[ID] = 0
    
    user_prob = lda_trained.get_document_topics(user_corpus, minimum_probability=None)
    for ID, prob in user_prob:
        if ID in selected_topic_id:
            selected_topic_prob[ID] = prob
    
    improve_topic_ids = [ID for ID in selected_topic_prob if selected_topic_prob[ID] < min_prob]  
    improve_topic_keywords = {}
    for i in improve_topic_ids:
        title = selected_topic_title[i]
        if title not in improve_topic_keywords:
            improve_topic_keywords[title] = ', '.join([v[0] for v in lda_trained.show_topic(i, Num_Words)])
        else:
            improve_topic_keywords[title] = ''
    print(lda_trained[user_corpus]) # get topic probability distribution for user review
    print(lda_trained.get_document_topics(user_corpus, minimum_probability=None))
    print(improve_topic_keywords)
    
    output_prob = []

    for ID, probi in selected_topic_prob.items():
        output_prob.append(probi)

    norm_output = [float(i)/sum(output_prob) for i in output_prob]
    return improve_topic_keywords,  norm_output

