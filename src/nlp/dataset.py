HF_MODEL_MAX_LENGTH = 512
from transformers import BertTokenizer as tokenizerlib
import datasets
import numpy as np
from src.data import get_distill_dataobj
import src.nlp.dataset as nlp_dataset
import src.nlp.model as nlp_model
import os
from os.path import join

hf_ds_config = {
    'agnews': {'ds_name': 'ag_news', 
               'num_labels': 4,
               'test_name': 'test',
               'data_name': 'text',
               'label_name': 'label'}, 

    'sst2': {'ds_name': 'stanfordnlp/sst2',
             'num_labels': 2,
             'test_name': 'validation',
             'data_name': 'sentence',
             'label_name': 'label'}, 

    'mnli': {'ds_name': 'nyu-mll/multi_nli',
             'num_labels': 3,
             'test_name': 'validation_matched',
             'data_name': ('premise', 'hypothesis'),
             'label_name': 'label'}, 

    'marc': {'ds_name': 'mteb/amazon_reviews_multi', 
             'num_labels': 5,
             'test_name': 'test',
             'data_name': 'text',
             'label_name': 'label',
             'val_name': 'validation'},
                                  
    'sent140': {'ds_name': 'sentiment140', 
                'num_labels': 2,
                'test_name': None,
                'data_name': 'text',
                'label_name': 'sentiment'},

    'bookcorpus': {'ds_name': 'bookcorpus', 
                   'num_labels': 0,
                   'test_name': None,
                   'data_name': 'text',
                   'label_name': None},

    'snli': {'ds_name': 'stanfordnlp/snli',
             'num_labels': 3,
             'test_name': 'test',
             'data_name': ('premise', 'hypothesis'),
             'label_name': 'label'},

    'dbpedia': {'ds_name': 'fancyzhx/dbpedia_14',
                'num_labels': 14,
                'test_name': 'test',
                'data_name': 'content',
                'label_name': 'label'},
    
    'yelp': {'ds_name': 'yelp_review_full',
             'num_labels': 5,
             'test_name': 'test',
             'data_name': 'text',
             'label_name': 'label'},
}

def resize_ds(ds, size, seed, label_name): 
    rng = np.random.default_rng(seed)
    size = min(size, len(ds))
    
    if label_name is None: 
        return ds.select(rng.choice(len(ds), size, replace=False).tolist())
    
    ds_labels = np.array(ds[label_name])
    unq_labels = np.unique(ds_labels)

    new_idx = []
    for label in unq_labels: 
        idx = np.where(ds_labels==label)[0]
        new_idx.append(rng.choice(idx, min(len(idx),int(size/len(unq_labels))), replace=False).tolist())
    new_idx = np.concatenate(new_idx)

    return ds.select(new_idx)

def prep_dataset(dataset, datadir, w_name, max_length, seed, size=None, concat=False): 
    test_name = hf_ds_config[dataset]['test_name']
    label_name = hf_ds_config[dataset]['label_name']
    data_name = hf_ds_config[dataset]['data_name']
    ds_name = hf_ds_config[dataset]['ds_name']

    dataset_dir = join(datadir,'datasets',ds_name)
    ds = {}
    for splitdir in os.listdir(dataset_dir): 
        if not os.path.isdir(join(dataset_dir, splitdir)): 
            continue
        ds[splitdir] = datasets.load_from_disk(join(dataset_dir, splitdir))

    if test_name is None: 
        test_size = 0.1
        ds = ds['train'].train_test_split(test_size=test_size)
        train_ds = ds['train']
        test_ds = ds['test']
    else: 
        train_ds = ds['train']
        test_ds = ds[test_name]
        if 'val_name' in hf_ds_config[dataset].keys(): 
            val_name = hf_ds_config[dataset]['val_name']
            test_ds = datasets.concatenate_datasets([test_ds, ds[val_name]])
    
    if dataset == 'sent140': 
        # Relabel 4 to 1
        train_ds = train_ds.map(
            lambda input: {label_name:0 if input[label_name] == 0 else 1}, 
            features=train_ds.features,
            load_from_cache_file=True
        )
        test_ds = test_ds.map(
            lambda input: {label_name:0 if input[label_name] == 0 else 1}, 
            features=test_ds.features,
            load_from_cache_file=True
        )
    elif dataset == 'marc': 
        # Get english reviews
        train_ds = train_ds.select(list(range(200000,400000)))
        test_ds = test_ds.select([*list(range(5000,10000)),*list(range(35000,40000))])
    elif dataset == 'snli': 
        trl, tel = train_ds[label_name], test_ds[label_name]
        train_ds = train_ds.select(
            [i for i in range(len(trl)) if trl[i] in [0,1,2]]
        )
        test_ds = test_ds.select(
            [i for i in range(len(tel)) if tel[i] in [0,1,2]]
        )

    tokenizer = tokenizerlib.from_pretrained(os.path.join(datadir,'tokenizers',w_name))
    tokenizer.model_max_length = max_length

    def tokenize_func(input): 
        if type(data_name) == tuple: 
            return tokenizer(*[input[data_name[_]] for _ in range(len(data_name))],
                            padding='max_length', 
                            truncation=True)
        else: 
            return tokenizer(input[data_name], 
                            padding='max_length', 
                            truncation=True)
    
    def tokenize_dataset(dataset): 
        tokenized_dataset = dataset.map(tokenize_func, batched=True, load_from_cache_file=True)
        valid_columns = ['input_ids', 'token_type_ids', 'attention_mask', label_name]
        columns_to_remove = [c for c in dataset.features.keys() if c not in valid_columns]
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        tokenized_dataset = tokenized_dataset.with_format('torch')
        if label_name is not None: 
            tokenized_dataset = tokenized_dataset.rename_column(label_name, 'labels')
        return tokenized_dataset

    if concat: 
        ds = datasets.concatenate_datasets([train_ds, test_ds])
        if size is not None: 
            ds = resize_ds(ds, size, seed, label_name)
        if label_name is None: 
            ds = ds.add_column('label', [-1]*len(ds))
            label_name = 'label'
            hf_ds_config['label_name'] = 'label'
            ds = ds.class_encode_column('label')
        return tokenize_dataset(ds)
    else: 
        if size is not None: 
            train_ds = resize_ds(train_ds, size, seed, label_name)
        return tokenize_dataset(train_ds), tokenize_dataset(test_ds)

def get_public_ds(args): 
    num_clusters = len(args.num_users)
    w_name = nlp_model.get_w_names(args.models)[0]
    num_classes = nlp_dataset.hf_ds_config[args.distill_dataset]['num_labels']
    public_ds=nlp_dataset.prep_dataset(
                    dataset=args.distill_dataset, 
                    datadir=args.datadir,
                    w_name=w_name, 
                    max_length=nlp_model.MAX_LENGTH, 
                    seed=args.seed,
                    size=args.public_size,
                    concat=True
                )
    public_ds = get_distill_dataobj(
                    alg=args.alg, 
                    train_ds_name=args.dataset,
                    num_clusters=num_clusters, 
                    public_ds=public_ds,
                    num_classes=num_classes,
                    nlp=args.nlp
                )
    return public_ds

def get_train_test_ds(args): 
    w_name = nlp_model.get_w_names(args.models)[0]
    return nlp_dataset.prep_dataset(
        dataset=args.dataset, 
        datadir=args.datadir,
        w_name=w_name, 
        max_length=nlp_model.MAX_LENGTH,
        seed=args.seed,
        size=args.train_size
    )
