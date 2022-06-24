import json
import time
import RecommenderAPI
import inference

def do_training():
    print('Load Sample data')
    f = open("recommender.json")
    print('Convert Sample data to JSON')
    data = json.load(f)
    print('Do Training')
    start_time = time.time()
    RecommenderAPI.do_training(data, "local_example")
    print("Training done in %s seconds" % (time.time() - start_time))
    
 
def do_prediction():
    print('Load Sample data')
    f = open("recommender.json")
    print('Convert Sample data to JSON')
    data = json.load(f)
    print('Select first item')
    concept_map = data[0]
    print('Do Prediction')
    start_time = time.time()
    result = inference.do_prediction(concept_map, "local_example")
    print("Prediction done in %s seconds" % (time.time() - start_time))
    print('Result is:', str(result))
 
#do_training()
do_prediction()