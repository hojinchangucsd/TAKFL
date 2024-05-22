from transformers import BertForSequenceClassification as modellib
import os

hf_weight_names = {
    'bert-tiny': 'prajjwal1/bert-tiny', 
    'bert-mini': 'prajjwal1/bert-mini', 
    'bert-small': 'prajjwal1/bert-small'
}

MAX_LENGTH = 512

def get_w_names(names): 
    return [hf_weight_names[n] for n in names]

def get_models(num_users, model, num_labels, datadir): 
    model_name = hf_weight_names[model]
    model_path = os.path.join(datadir,'models',model_name)
    netglob = modellib.from_pretrained(model_path, 
                                       num_labels=num_labels,
                                       ignore_mismatched_sizes=True,
                                       local_files_only=True).cpu()
    netglob.classifier.weight.data.normal_()
    init_state_dict = netglob.state_dict()
    user_models = []
    for _ in range(num_users): 
        user_model = modellib(netglob.config).cpu()
        user_model.load_state_dict(init_state_dict)
        user_models.append(user_model)
    return user_models, netglob, init_state_dict
