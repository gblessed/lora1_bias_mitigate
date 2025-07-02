import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
print(os.getcwd())
from models.model import Model

model_id = "microsoft/phi-2"
phi_2_model = Model(model_id)

from custom_datasets.Dataset import CustomDataset

dataset = CustomDataset(dataset_type='teleqna', model_type='microsoft/phi-2', tokenizer=phi_2_model.tokenizer)

from utils.data_utils import build_dataloader
k_limit = 5
new_data_loader = build_dataloader(batch_size=1,shuffle=False,custom_dataset=dataset,collate_fn=dataset.custom_collate_fn(k_limit))

sample_question = None
for batch in new_data_loader:
    print(len(batch))
   
    