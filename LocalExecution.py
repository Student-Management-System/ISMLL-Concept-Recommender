import json
import time
import concept_map_recommender_training_api
import concept_map_recommender_inference_api
import group_recommender_batch_api

def do_training():
    print('Load Sample data')
    f = open("testdata/concept_map_export.json")
    print('Convert Sample data to JSON')
    data = json.load(f)
    print('Do Training')
    start_time = time.time()
    concept_map_recommender_training_api.do_training(data, "local_example")
    print("Training done in %s seconds" % (time.time() - start_time))
    
 
def do_prediction():
    print('Load Sample data')
    f = open("testdata/concept_map_recommendation_query.json")
    print('Convert Sample data to JSON')
    concept_map = json.load(f)
    print('Do Prediction')
    start_time = time.time()
    result = concept_map_recommender_inference_api.do_prediction(concept_map, "local_example")
    print("Prediction done in %s seconds" % (time.time() - start_time))
    print('Result is:', str(result))

def group_nn_train_and_predict():
    print('Load Sample data from file')
    f = open('testdata/recommender-export.json')
    print('Convert Sample data to JSON')
    course_export = json.load(f)
    print('Do Batch Analysis')
    start_time = time.time()
    result = group_recommender_batch_api.group_recommendation_with_neuronal_network(course_export)
    print("Prediction done in %s seconds" % (time.time() - start_time))
    print('Result is:', str(result))
   
def group_knearest_train_and_predict():
    print('Load Sample data from file')
    f = open('testdata/recommender-export.json')
    print('Convert Sample data to JSON')
    course_export = json.load(f)
    print('Do Batch Analysis')
    start_time = time.time()
    result = group_recommender_batch_api.group_recommendation_with_knearest_neighbours(course_export)
    print("Prediction done in %s seconds" % (time.time() - start_time))
    print('Result is:', str(result))
 
#do_training()
do_prediction()