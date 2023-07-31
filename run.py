import hydra
from omegaconf import DictConfig
import logging
import os
from easydict import EasyDict

from model_pipeline.model import Model
from model_pipeline.train import train, evaluate
from utils.set_seeds import set_seed

import torch
from transformers import (WEIGHTS_NAME, RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)                        
                        
@hydra.main(config_path="config", config_name="config", version_base='1.2',)
def main(args: EasyDict) -> None:  
    # Set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = str(device)
    logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)
    
    # Seeds everything
    set_seed(args.seed)


    # Build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)   
    model = Model(model)  # add pooling & LN based on the pretrained model
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    
    # Start training/evaluating process on the corresponding dataset
    logger.info("Training/evaluation parameters %s", args)
    
    # Training (if it is required)
    if args.dataset['do_train']:
        train(args, model, tokenizer)
        
    # Evaluating/Testing 
    results = {}
    if args.dataset['do_eval']:
        evaluate_and_log_results(args, model, tokenizer, args.dataset['eval_data_file'])    
    if args.dataset['do_test']:
        evaluate_and_log_results(args, model, tokenizer, args.dataset['test_data_file'])


def evaluate_and_log_results(args, model, tokenizer, data_file):
    if args.dataset.do_zero_shot is False:  # evaluation based on the best paras we have
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.dataset.output_dir, checkpoint_prefix)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    result = evaluate(args, model, tokenizer, data_file)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 3)))

if __name__ == "__main__":
    main()