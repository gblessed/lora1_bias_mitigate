from string import Template


def get_prompt_template(dataset_type, model_type):

    if  dataset_type == "medmcqa" and model_type in ["Qwen/Qwen2.5-3B-Instruct" ,"microsoft/phi-2", "google-t5/t5-3b"] :

        prompt_without_contex_train= Template('''Instruct = Youre a Medical Question Answering Expert, answer the following question. Please generate only answer choice (1, 2, 3 or 4)\n                                                                 
        $question
        $options
        Output: option ''')

    elif dataset_type == "teleqna" and model_type in ["Qwen/Qwen2.5-3B-Instruct","google-t5/t5-3b"]  :
    
        prompt_without_contex_train= Template('''Instruct: $question
        Abbreviations: $abbreviation
                
        Considering the following contexts:
        context 1: $context1
        context 2: $context2
        context 3: $context3      
                                                                            
        $question
        $options
        Output: option ''')

    elif dataset_type == "teleqna" and model_type == "microsoft/phi-2":

        prompt_without_contex_train= Template('''Instruct: $question
        Abbreviations: $abbreviation
                
        Considering the following contexts:
        context 1: $context1
        context 2: $context2
        context 3: $context3      
                                                                            
        $question
        $options
        Output: option ''')

    elif dataset_type == "qasc" and model_type in ["Qwen/Qwen2.5-3B-Instruct","google-t5/t5-3b" ,"microsoft/phi-2"]:

        prompt_without_contex_train = Template('''Instruct: Answer the following question using the context provided, reason over it because only one of the context is relevant . Please generate only answer choice (1, 2, 3, 4, 5, 6, 7 or 8) without any explanations\n
        $question
        context: $context      
                                                                
        $options
        $question
        Output: option 
        ''')
    else:
        return None
    
    return prompt_without_contex_train