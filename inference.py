

def predict(student_map, model, concept_to_idx, idx_to_concept):
    """This function recommend concepts to students
    
       param: json_file, trained model, concepts dictionary
       return: Top recommended concepts.
    """
    
    
    list_concepmaps =list()
    for i in student_map:  
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
    

####Example: how to call the inference function
top_concepts = predict(student_map, model, concept_to_idx, idx_to_concept)

