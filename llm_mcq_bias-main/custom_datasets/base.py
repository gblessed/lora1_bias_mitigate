
import json

def load_json_file(data_type: str, data_path: str):
    """
    Load JSON data from a file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        list or dict: Parsed JSON data.
    """
    data_map = {}
    data = []
    if data_type == "medmcqa" or data_type == "qasc":
        
        with open(data_path+'l', "r") as f:
            test_data = f.readlines()

        for i, line in enumerate(test_data):
            data.append(json.loads(line))
            data_map[i] =  i
        return data
    
    elif data_type == "teleqna" :
        with open(data_path, "r") as f:
            data = json.load(f)
        for i , key  in  enumerate(list(data.keys())):
            data_map[i] = key
    return data, data_map


    

def preprocess(text):
    
    text = text.lower()
    return set(text.split())


def choose_most_likely_option(options, candidate_answer):
    candidate_words = preprocess(candidate_answer)
    
    best_option = None
    max_overlap = 0
    
    for option in options:
        option_words = preprocess(option)
        overlap = len(candidate_words.intersection(option_words))
        
  
        if overlap > max_overlap:
            max_overlap = overlap
            best_option = option
            
    return best_option, options.index(best_option)+1