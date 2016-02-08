
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import gensim


# In[ ]:

def ldaReviewModel_Test_USER(user_review_texts):
    
    userReview = user_review_texts
    num_of_reviews = len(userReview)
    
    ## parameter
    Num_Words = 15
    Num_Topics = 20
    Num_Selected_Topics = 8
    selected_topic_id = [2,9,11,12,14,15,17,18]
    num_selected_topics = len(selected_topic_id)
    selected_topic_title = {2:'Communication', 9:'Transportation',11:'Neighbourhood:Convenience',
                            12:'Experience',14:'Arrival',15:'Neighbourhood:Fun',17:'Food',18:'Amenity'}
    min_prob = 0
    
    ## Proprocess review data: Parameters
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = get_stop_words('en')
    ## high frequency words
    custom_stop = ['stay','great','place','us','love','like','host','recommend','everything','san','francisco',
                   'perfect','location','apartment', 'nice','good','definitely','really','room','house', 'didn',
                   'also','just','even','well','make','will','much','airbnb']

    single_letters = [let for let in 'abcdefghijklmnopqrstuvwxyz']
    stop_words_list = en_stop + custom_stop
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # Create lemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    
    ## Load LDA
    lda_trained = models.LdaModel.load('lda20_3.model')
    topics = lda_trained.show_topics(num_topics=Num_Topics, num_words=Num_Words, formatted=False)
    topics_table = {}
    for ti, topic in enumerate(topics):
        topics_table[ti] = ', '.join([v[0] for v in lda_trained.show_topic(ti, Num_Words)])
    
    result = [[0 for i in range(Num_Selected_Topics)] for j in range(num_of_reviews)]
    combined_result = [0]*Num_Selected_Topics
    # loop through document list
    for k, rev in enumerate(userReview):
        # list for tokenized documents in loop
        user_texts = []
        # clean and tokenize document string
        user_raw = str(rev).lower()
        user_tokens = tokenizer.tokenize(user_raw)

        # remove stop words from tokens
        user_stopped_tokens = [i for i in user_tokens if not i in en_stop]
    
        # lemmatize tokens
        user_lemmatized_tokens_noun = [wordnet_lemmatizer.lemmatize(i) for i in user_stopped_tokens]
        #lemmatized_tokens_adj = [wordnet_lemmatizer.lemmatize(i, wn.ADJ) for i in lemmatized_tokens_noun]
        user_lemmatized_tokens_verb = [wordnet_lemmatizer.lemmatize(i, wn.VERB) for i in user_lemmatized_tokens_noun]
        # stem tokens
        # stemmed_tokens = [p_stemmer.stem(i) for i in lemmatized_tokens]
        # fillter high freq words
        user_filtered_tokens = [i for i in user_lemmatized_tokens_verb if len(i) > 3 and not i in custom_stop]
        # add tokens to list
        user_texts.extend(user_filtered_tokens)
    
        # turn our tokenized documents into a id <-> term dictionary
        user_dictionary = corpora.Dictionary([user_texts])
        # convert tokenized documents into a document-term matrix
        user_corpus = user_dictionary.doc2bow(user_texts)
    
        ## Apply LDA on user reviews
        selected_topic_prob = {}
        for ID in selected_topic_id:
            selected_topic_prob[ID] = 0
    
        user_prob = lda_trained.get_document_topics(user_corpus, minimum_probability=None)
        for ID, prob in user_prob:
            if ID in selected_topic_id:
                selected_topic_prob[ID] = prob
        output_prob = []

        for ID, probi in selected_topic_prob.items():
            output_prob.append(probi)
        norm_output = []
        norm_output = [float(i)/sum(output_prob) for i in output_prob]
        for j in range(Num_Selected_Topics):
            result[k][j] = norm_output[j]
    
    for topic in range(Num_Selected_Topics):
        score = 0
        for re in range(num_of_reviews):
            score += result[re][topic]
            
        combined_result[topic] = score/num_of_reviews
    
    final_selected_topic_prob = {}
    for i,ID in enumerate(selected_topic_id):
        final_selected_topic_prob[ID] = combined_result[i]
    
    improve_topic_ids = [ID for ID in final_selected_topic_prob if final_selected_topic_prob[ID] < min_prob]  
    improve_topic_keywords = {}
    for i in improve_topic_ids:
        title = selected_topic_title[i]
        if title not in improve_topic_keywords:
            improve_topic_keywords[title] = ', '.join([v[0] for v in lda_trained.show_topic(i, Num_Words)])
        else:
            improve_topic_keywords[title] = ''
    #print(lda_trained[user_corpus]) # get topic probability distribution for user review
    #print(lda_trained.get_document_topics(user_corpus, minimum_probability=None))
    #print(improve_topic_keywords)
    
    
    return improve_topic_keywords,  combined_result

