import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class LambdaWrapper(torch.nn.Module):
    def __init__(self, num_clusters, alpha_init, w_avg, task_vectors, arch, names):
        super(LambdaWrapper, self).__init__()
        if alpha_init is None: 
            alpha_init = F.softmax(torch.ones((num_clusters,)).cpu(), dim=0)
        else: 
            alpha_init = [math.log(a) for a in alpha_init]
            alpha_init = torch.tensor(alpha_init).cpu()
        self.raw_alphas = torch.nn.Parameter(alpha_init)
        self.beta = torch.nn.Parameter(torch.ones((1,)).cpu())
        self.w_avg = w_avg
        self.task_vectors = task_vectors
        self.arch = arch
        self.names = names
        self.device = torch.device('cpu')

    def to(self, device): 
        self.device = device
        self = super().to(device)
        self.w_avg = tuple(w.to(device) for w in self.w_avg)
        self.task_vectors = [tuple(v.to(device) for v in tv) for tv in self.task_vectors]
        self.arch = self.arch.to(device)

    def alphas(self): 
        return F.softmax(self.raw_alphas, dim=0)
    
    def forward(self, data): 
        alphas = self.alphas()
        params = tuple(sum(tuple(p * alphas[j] for j,p in enumerate(tv))) + self.w_avg[i] for i, tv in enumerate(zip(*self.task_vectors)))
        load_weights(self.arch, self.names, params)
        out = self.arch(**data).logits if isinstance(data, dict) else self.arch(data)
        return self.beta * out
    
def train(wrapper, optimizer, dataset, epochs, batch_size, num_iters=None, verbose=False, verbose_freq=50): 
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    loss_func = torch.nn.CrossEntropyLoss()
    
    wrapper.train()
    for _ in range(epochs): 
        for i, data in enumerate(dl): 

            if num_iters is not None and num_iters <= i: 
                break

            if isinstance(data, dict): 
                labels = data['labels'].to(wrapper.device)
                data_in = {k: v.to(wrapper.device) for k, v in data.items() if k != 'labels'}
            else: 
                data_in, labels = data
                data_in, labels = data_in.to(wrapper.device), labels.to(wrapper.device)
            
            wrapper.zero_grad()
            optimizer.zero_grad()
            
            out = wrapper.forward(data_in)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            str_alpha_list = [f'{alpha:.4f}' for alpha in wrapper.alphas().tolist()]
            if verbose and i % verbose_freq == 0: 
                print(f'Tune Lambda iter {i} -- Alphas {str_alpha_list} -- Loss {loss:.4f}')
            
    return wrapper.alphas().tolist()