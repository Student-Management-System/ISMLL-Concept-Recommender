import torch
import numpy as np
import pickle


PAD = 0
MASK = 1

def do_prediction(student_map, service_name):
    model = load('model', service_name)
    concept_to_idx = load('concept_to_idx', service_name)
    idx_to_concept = load('idx_to_concept', service_name)

    return predict(student_map, model, concept_to_idx, idx_to_concept)


def predict(student_map, model, concept_to_idx, idx_to_concept):
    
    d = open(student_map)
    testdata = json.load(d)
    
    list_concepmaps =list()
    for i in testdata:
        for f in i['concepts']:
            list_concepmaps.append(f['name'])
            
    ids = [PAD] * (120 - len(list_concepmaps) - 1) + [concept_to_idx[a] for a in list_concepmaps] + [MASK]
    
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(src)
    
    masked_pred = prediction[0, -1].numpy()
    
    sorted_predicted_ids = np.argsort(masked_pred).tolist()[::-1]
    
    sorted_predicted_ids = [a for a in sorted_predicted_ids if a not in ids]
    
    return [idx_to_concept[a] for a in sorted_predicted_ids[:10] if a in idx_to_concept]


def load(file_name, folder):
    with open('models/' + folder + '/' + file_name, "rb") as infile:
        return pickle.load(infile)     

####Example: how to call the inference function
# top_concepts = predict('test.json', model, concept_to_idx, idx_to_concept)


