#%%
from mammoth_lite import train, load_runner

#%%
args, model, dataset = load_runner('sgd','seq-cifar10',{'lr': 0.1, 'n_epochs': 1, 'batch_size': 32})

#%%
train(model, dataset, args)

#%%