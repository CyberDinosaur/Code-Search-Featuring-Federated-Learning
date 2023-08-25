import numpy as np
import logging
import torch
import wandb
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, random_split, RandomSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup)

logger = logging.getLogger(__name__)                        

class client(object):
    def __init__(self, local_dataset, dev):
        self.train_ds = local_dataset
        self.train_dl = None
        self.dev = dev
        self.local_parameters = None

    def localUpdate(self, model, local_epochs, local_batchsize, learning_rate, max_grad_norm, mu, global_parameters, isWANDB, method):
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
                
                if method == 'FedAvg':
                    total_loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))
                    
                elif method == 'FedProx':
                    original_loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))
                    proximal_term = 0.0
                    for param_name, param in model.named_parameters():
                        # Make sure we don't add proximal term for bias terms
                        if 'bias' not in param_name:
                            global_param = global_parameters[param_name]
                            proximal_term += torch.sum((param - global_param).pow(2))
                    proximal_term *= mu / 2
                    total_loss = original_loss + proximal_term
                    
                elif method == 'FedNova':
                    original_loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))
                    momentum_term = 0.0
                    for param_name, param in model.named_parameters():
                        global_param = global_parameters[param_name]
                        momentum_term += torch.sum((param - global_param).pow(2))
                    momentum_term *= mu
                    total_loss = original_loss + momentum_term

                # Backward
                total_loss.backward()

                # Report loss
                tr_num, tr_loss = 0, 0
                tr_loss += total_loss.item()
                tr_num += 1
                if (step + 1) % 100 == 0:
                    logger.info("epoch {} step {} loss {}".format(epoch, step+1, round(tr_loss/tr_num,5)))
                    if isWANDB:
                        wandb.log({"loss": round(tr_loss/tr_num, 5)})
                    tr_loss = 0
                    tr_num = 0
                
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



