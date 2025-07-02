import numpy as np
from typing import List

from sklearn.metrics import recall_score

def fluctuation_rate(forward: List[int], backward: List[int]) -> float:
    '''
    Parameters:
    forward (list or np.array): Predictions from the forward pass.
    backward (list or np.array): Predictions from the backward pass.
    '''
    # Ensure both inputs have the same length
    assert len(forward) == len(backward), "The lengths of forward and backward must be the same"
    
    differing_predictions = np.sum(np.array(forward) != np.array(backward))
    
    return differing_predictions / len(forward)

'''
Example 

forward = [1, 0, 1, 0, 1]
backward = [1, 1, 1, 0, 0]
print(fluctuation_rate(forward, backward)) # gives 0.4

'''



def rstd(predictions: List[int], ground_truth: List[int]) -> float:
    '''
    Parameters:
    predictions (list or np.array): Predictions 
    ground_truth (list or np.array): Ground truth
    '''
    # Ensure both inputs have the same length
    assert len(predictions) == len(ground_truth), "The lengths of predictions and ground_truth must be the same"
    predictions =  np.array(predictions)
    ground_truth =  np.array(ground_truth)
    class_wise_recalls = []
    for cls in np.unique(ground_truth):
        indices = np.where(ground_truth == cls)
        ground_truth_cls = ground_truth[indices]
        predictions_cls = predictions[indices]

        class_wise_recalls.append(recall_score(ground_truth_cls,predictions_cls, pos_label=cls, zero_division=0, ))






        print(cls,ground_truth_cls, predictions_cls)     
    return np.std(class_wise_recalls)


'''
Example 

predictions = [1, 1, 0, 0, 0]
ground_truth = [1, 0, 1, 0, 1]
rstd(predictions, ground_truth)# gives 0.0833333
'''


def ckld(predictions: List[int], ground_truth: List[int]) -> float:
    '''
    Parameters:
    predictions (list or np.array): Predictions
    ground_truth (list or np.array): Ground truth
    '''
    # Ensure both inputs have the same length
    assert len(predictions) == len(ground_truth), "The lengths of predictions and ground_truth must be the same"
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    all_classes = np.unique(ground_truth)

    gt_counts = np.bincount(ground_truth, minlength=len(all_classes))
    pred_counts = np.bincount(predictions, minlength=len(all_classes))


    p_cls = gt_counts / len(ground_truth)
    q_cls = pred_counts / len(predictions)

    mask = (p_cls > 0) & (q_cls > 0)
    p_cls = p_cls[mask]
    q_cls = q_cls[mask]
    
  
    divergence = np.sum(p_cls * np.log(p_cls / q_cls))
    
    return divergence

'''
predictions = [1, 1, 0, 0, 0]
ground_truth = [1, 0, 1, 0, 1]
ckld(predictions, ground_truth)
'''