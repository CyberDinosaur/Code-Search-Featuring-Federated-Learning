import logging
import os
import torch
import numpy as np
import tqdm
import wandb

from model_pipeline.data import TextDataset
from model_pipeline.clients import ClientsGroup

from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup)

logger = logging.getLogger(__name__)                        

def distributed_train(args, model, tokenizer):
    """ Train the model in a distributed manner"""
    # Get training dataset
    train_dataset = TextDataset(tokenizer, args, args.dataset['train_data_file'])
       
    # Create Clients Group
    myClients = ClientsGroup(train_dataset, args.training['isIID'], args.training['num_of_clients'], args.device)
    num_in_comm = int(max(args.training['num_of_clients'] * args.training['cfraction'], 1))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num total examples = %d", len(train_dataset))
    logger.info("  Num Clients = %d", args.training['num_of_clients'])
    logger.info("  Num communications between Server and Clients = %d", num_in_comm)
    logger.info("  Train batch size for each client = %d", args.train_batch_size)
    model.zero_grad()
    model.train()
    

    # Create Global parameters and Update global parameters distributively
    global_parameters = {}
    for key, var in model.state_dict().items():
        global_parameters[key] = var.clone()
    
    # Initialize control params (only need in SCAFFOLD)
    if args.training['method_name'] == 'SCAFFOLD':
        control_parameters = {}
        for key, var in model.state_dict().items():
            control_parameters[key] = torch.zeros_like(var)

        
    for i in range(args.training['num_comm']):
        logger.info("Communicate round {}".format(i+1))
        # Pick clients_in_comm
        order = np.random.permutation(args.training['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        # Assign malicious clients
        num_malicious = int(np.ceil(args.training['malicious_prob'] * len(clients_in_comm)))
        # malicious_clients = np.random.choice(clients_in_comm, num_malicious, replace=False)
        malicious_clients = ['client0']
        logger.info(" Malicious clients: {}".format(malicious_clients))

        sum_parameters = None
        sum_control = None if args.training['method_name'] == 'SCAFFOLD' else None

        # Local update
        sum_parameters = None
        for client in (clients_in_comm):
            logger.info("   {} running now!".format(client))

            if args.training['method_name'] == 'FedAvg':
                local_parameters = myClients.clients_set[client].localUpdate(
                    model, args.dataset['num_train_epochs'], args.train_batch_size,
                    args.learning_rate, args.max_grad_norm, args.training['mu'], global_parameters,
                    args.isWANDB, args.training['method_name'])
            else:  # SCAFFOLD
                local_parameters, local_control = myClients.clients_set[client].localUpdate(
                    model, args.dataset['num_train_epochs'], args.train_batch_size,
                    args.learning_rate, args.max_grad_norm, args.training['mu'], global_parameters,
                    control_parameters, args.isWANDB, args.training['method_name'])

            # After local update, check if this client is malicious.
            if client in malicious_clients:
                logger.info(f"{client} is malicious!")
                
                for key in local_parameters.keys():
                    local_parameters[key] *= -1.0


            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
                
                if args.training['method_name'] == 'SCAFFOLD':
                    sum_control = {}
                    for key, var in local_control.items():
                        sum_control[key] = var.clone()
            else:
                for key in sum_parameters.keys():
                    sum_parameters[key] += local_parameters[key]
                    
                    if args.training['method_name'] == 'SCAFFOLD':
                        sum_control[key] += local_control[key]

            # local_parameters = myClients.clients_set[client].localUpdate(
            #     model, args.dataset['num_train_epochs'], args.train_batch_size, 
            #     args.learning_rate, args.max_grad_norm, args.training['mu'], global_parameters, 
            #     args.isWANDB, args.training['method_name'])
            # if sum_parameters is None:
            #     sum_parameters = {}
            #     for key, var in local_parameters.items():
            #         sum_parameters[key] = var.clone()
            # else:
            #     for var in sum_parameters:
            #         sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        # Global update
        if args.training['method_name'] == 'FedAvg':
            for key in global_parameters.keys():
                global_parameters[key] = sum_parameters[key] / num_in_comm
        else:  # SCAFFOLD
            for key in global_parameters.keys():
                global_parameters[key] -= control_parameters[key]
                global_parameters[key] += sum_parameters[key] / num_in_comm
                control_parameters[key] = sum_control[key] / num_in_comm
        # for var in global_parameters:
        #     global_parameters[var] = (sum_parameters[var] / num_in_comm)

        # Evaluate after every round of distributed training
        best_mrr = 0
        with torch.no_grad():        
            results = evaluate(args, model, tokenizer, args.dataset['eval_data_file'], eval_when_training=True)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value,4))     
                 
            if args.isWANDB:
                wandb.log({"evaluate_mrr/round": results['eval_mrr']})   
                
            # Save the best model
            if results['eval_mrr'] > best_mrr:
                best_mrr = results['eval_mrr']
                logger.info("  " + "*" * 20)  
                logger.info("  Best mrr:%s",round(best_mrr,4))
                logger.info("  " + "*" * 20)                          

                checkpoint_prefix = 'checkpoint-best-mrr'
                output_dir = os.path.join(args.dataset['output_dir'], '{}'.format(checkpoint_prefix))                        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)                        
                model_to_save = model.module if hasattr(model,'module') else model
                output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                torch.save(model_to_save.state_dict(), output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)


def train(args, model, tokenizer):
    """ Train the model """
    # Get training dataset
    train_dataset = TextDataset(tokenizer, args, args.dataset['train_data_file'])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    
    # Get optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.dataset['num_train_epochs'])

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.dataset['num_train_epochs'])
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.dataset['num_train_epochs'])
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0 
    for idx in range(args.dataset['num_train_epochs']): 
        for step, batch in enumerate(train_dataloader):
            # Get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            # Get code and nl vectors
            code_vec = model(code_inputs = code_inputs)
            nl_vec = model(nl_inputs = nl_inputs)   
            # Calculate scores and loss
            scores = torch.einsum("ab, cb->ac", nl_vec, code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))
            # Report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step+1, round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        # Evaluate    
        results = evaluate(args, model, tokenizer, args.dataset['eval_data_file'], eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        # Save best model
        if results['eval_mrr'] > best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  " + "*" * 20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  " + "*" * 20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.dataset['output_dir'], '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, file_name, eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, args.dataset['codebase_file'])
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    
    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())  

    model.train()    
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    scores = np.matmul(nl_vecs, code_vecs.T)  # num_queries * num_codes
    
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)      
    for example in code_dataset.examples:
        code_urls.append(example.url)

    # For every query, calculate its rank reciprocal
    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result