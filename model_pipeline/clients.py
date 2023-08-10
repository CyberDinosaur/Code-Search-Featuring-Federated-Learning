import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, random_split, RandomSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup)

class client(object):
    def __init__(self, local_dataset, dev):
        self.train_ds = local_dataset
        self.train_dl = None
        self.dev = dev
        self.local_parameters = None

    def localUpdate(self, model, local_epochs, local_batchsize, learning_rate, max_grad_norm, global_parameters):
        """ Update local parameters"""
        model.load_state_dict(global_parameters, strict=True)  # init local model according to serve's parameters
        train_sampler = RandomSampler(self.train_ds)
        self.train_dl = DataLoader(self.train_ds, sampler=train_sampler, batch_size=local_batchsize)
        
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(self.train_dl) * local_epochs)

        for epoch in range(local_epochs):
            for step, batch in enumerate(self.train_dl):
                # Get inputs
                code_inputs = batch[0].to(self.dev)    
                nl_inputs = batch[1].to(self.dev)
                # Get code and nl vectors
                code_vec = model(code_inputs = code_inputs)
                nl_vec = model(nl_inputs = nl_inputs)   
                # Calculate scores and loss
                scores = torch.einsum("ab, cb->ac", nl_vec, code_vec)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        return model.state_dict()


class ClientsGroup(object):
    def __init__(self, full_dataset, isIID, num_clients, dev):
        self.full_dataset = full_dataset
        self.is_iid = isIID
        
        self.num_of_clients = num_clients
        self.dev = dev
        self.clients_set = {}

        self.createClients()

    def createClients(self):
        """Distribute the full dataset to clients"""
        num_subdatasets = self.num_of_clients
        subdatasets_size = len(self.full_dataset) // num_subdatasets
        remainder = len(self.full_dataset) % num_subdatasets

        if self.is_iid:
            subdataset_sizes = [subdatasets_size] * num_subdatasets
            # Distribute remaining data randomly among subdatasets
            for i in range(remainder):
                subdataset_sizes[i] += 1

            subdatasets = random_split(self.full_dataset, subdataset_sizes)
        else:
            pass

        for i, subdataset in enumerate(subdatasets):
            someone = client(subdataset, self.dev)
            self.clients_set['client{}'.format(i)] = someone



