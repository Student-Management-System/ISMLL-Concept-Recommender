# Based on:
# * https://www.imaginarycloud.com/blog/flask-python/
# * https://realpython.com/api-integration-in-python/#flask

# Modules needed for web services
from flask import Flask, request
import json, jsonschema
from jsonschema import validate

# Modules needed for recommendation
import numpy as np 
import pandas as pd 
import os
import nltk
import gensim
from scipy.sparse import csr_matrix
#from IPython.display import display_html
import warnings
warnings.filterwarnings("ignore")

from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
from skopt import forest_minimize

# Moules needed for storing/loading local data
import pickle
from pathlib import Path

app = Flask(__name__)

@app.route('/concept_recommender/hello_world')
def hello_world():
    return 'Hello, World!'

def validate_json(json_data, spec):
    with open(spec, 'r') as file:
        api_schema = json.load(file)

    try:
        validate(instance=json_data, schema=api_schema)
    except jsonschema.exceptions.ValidationError as err:
        err = "Given JSON data is InValid"
        return False, err

    message = "Given JSON data is Valid"
    return True, message

def do_training_without_webservice():
    f = open("recommender.json")
    data = json.load(f)
    do_training(data, "local_example")

def do_training(concept_maps_as_json, service_name):
    data = convert_json_to_df(concept_maps_as_json)
    item_dict, embeddings = item_dictionary(data)
    books_metadata_csr = concept_embeddings(data)
    conceptmap_concept_interaction, conceptmap_concept_interaction_csr, concept_dict = concepts_interactions(data)
    model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.90,
                no_components=150,
                user_alpha=0.000005)

    model = model.fit(conceptmap_concept_interaction_csr,
                      epochs=100,
                      num_threads=16, verbose=False)
    
    # Save the models
    folder = 'models/' + service_name
    Path(folder).mkdir(parents=True, exist_ok=True)
    store(model, "model", folder)
    store(conceptmap_concept_interaction, "conceptmap_concept_interaction", folder)
    store(concept_dict, "concept_dict", folder)
    store(item_dict, "item_dict", folder)
    store(books_metadata_csr, "books_metadata_csr", folder)

def store(data, file_name, folder):
    with open(folder + '/' + file_name, "wb") as outfile:
        # "wb" argument opens the file in binary mode
        pickle.dump(data, outfile)

def convert_json_to_df(data):
    """This function convert JSON to DataFrame
    
       param: json_file:

       return: Final Dataframe.
    """
    
    conceptmap, users, conceptmaps_list, users_list, final_list, final_list2 = ([] for i in range(6))
    for i in data:
        d = i['conceptmap_id']
        s = i['user_id']
        conceptmap.append(d)
        users.append(s)
    my_list = ['map_id'] * len(conceptmap)
    my_list2 = ['user_id'] * len(conceptmap)
    for f, b in zip(conceptmap, my_list):
        f = [f]
        b = [b]
        intial = dict(zip(b, f))
        conceptmaps_list.append(intial) 
    for f, b in zip(users, my_list2):
        f = [f]
        b = [b]
        intial = dict(zip(b, f))
        users_list.append(intial)
    for f, b in zip(conceptmaps_list, data):
        i = b['concepts']
        result = [dict(item, **f) for item in i]
        for f in result:
            final_list.append(f) 
    for f, b in zip(users_list, data):
        i = b['concepts']
        result = [dict(item, **f) for item in i]
        for f in result:
            final_list2.append(f)
    df1 = pd.DataFrame(final_list)
    df2 = pd.DataFrame(final_list2)
    result = pd.concat([df1, df2], axis=1)
    result = result.loc[:,~result.columns.duplicated()].copy()
    result = result.drop(columns=[ 'timestamp'])
    result.rename(columns={'id': 'conceptId', 'name': 'title',  'map_id' :'ConceptMapID'}, inplace=True)
    data = result[['conceptId', 'ConceptMapID', 'title']]
    data.rename(columns={'title':'concepts'}, inplace=True)
    data['idx'] = data.groupby(['conceptId'], sort=False).ngroup()
    data = data.drop('conceptId', 1)
    data = data.rename(columns={'idx': 'conceptId'})
    return data
    
def item_dictionary(dataframe):
    """This function convert Dataframe to item dictionary
    
       param: Dataframe:

       return: item Dictionary.
    """
    url = "GoogleNews-vectors-negative300.bin"
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(url, binary=True)
    item_dict ={}
    df = dataframe[['conceptId', 'concepts']].sort_values('conceptId').reset_index()
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,'conceptId'])] = df.loc[i,'concepts']
    return item_dict, embeddings
    
def concept_embeddings(data, language='english'):
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    stopwords = nltk.corpus.stopwords.words(language) # removing stop words
    item_dict, embeddings = item_dictionary(data)
    for doc in data['concepts'].str.lower().str.replace('[^a-z ]', ''): # looping through each document and cleaning it
        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for word in doc.split(' '): # looping through each word of a single document and spliting through space
            if word not in stopwords: # if word is not present in stopwords then (try)
                try:
                    word_vec = embeddings[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                    temp = temp.append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe
                except:
                    pass
        doc_vector = temp.mean() # take the average of each column(w0, w1, w2,........w300)
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
        docs_vectors=docs_vectors.iloc[:, : 236]
        docs_vectors.fillna(0)    
        books_metadata_csr = csr_matrix(docs_vectors.values)# convert embeddings to csr matrix
    
    return books_metadata_csr
        
def concepts_interactions(data):
    data['rating'] = 10
    data.rename(columns={'conceptId':'concepts_id'}, inplace=True)
    unique_vals = data['concepts_id'].unique()
    data['concepts_id'].replace(to_replace=unique_vals, value= list(range(len(unique_vals))),inplace=True)
    final = data[['concepts_id', 'ConceptMapID', 'rating']]
    conceptmap_concept_interaction = pd.pivot_table(final, index='ConceptMapID', columns='concepts_id', values='rating')
    conceptmap_concept_interaction = conceptmap_concept_interaction.fillna(0)

    conceptmap_concept_interaction.head(20)
    conceptmap_id = list(conceptmap_concept_interaction.index)
    concept_dict = {}
    counter = 0 
    for i in conceptmap_id:
        concept_dict[i] = counter
        counter += 1
    conceptmap_concept_interaction_csr = csr_matrix(conceptmap_concept_interaction.values)
    return conceptmap_concept_interaction, conceptmap_concept_interaction_csr, concept_dict

@app.post('/concept_recommender/train')
def train():
    if request.is_json:
        data = request.get_json()
        valid, msg = validate_json(data, 'Multiple-Concept-Maps-Recommender.spec.json')

        if not valid:
            return {"Error": "JSON request does not represent Concept Map(s):\n" + msg}, 415 # 415 means Unsupported media type

        # Do further stuff with json data
        return "Model updated", 201 # 201 means something was created
    return {"Error": "Request must be JSON"}, 415 # 415 means Unsupported media type
    
    
do_training_without_webservice()