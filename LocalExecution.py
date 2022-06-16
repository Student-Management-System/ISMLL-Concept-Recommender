import json
import RecommenderAPI
import time

def do_training():
    print('Load Sample data')
    f = open("recommender.json")
    print('Convert Sample data to JSON')
    data = json.load(f)
    print('Do Training')
    start_time = time.time()
    RecommenderAPI.do_training(data, "local_example")
    print("Training done in %s seconds" % (time.time() - start_time))
    
    
do_training()