import argparse
import datetime
import glob
import logging
import os
import time

import torch

from logging_helper import init_logger
from models import Discriminator, BartSystem
from train import train
from transformer_base import add_generic_args, generic_train


class Config():
    # data_path = './data/chatbot/'
    # log_dir = 'runs/exp'
    save_path = './save'
    # pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 1024 # max_source_length
    # embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    # batch_size = 64
    lr_F = 5e-6
    lr_D = 1e-4
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 1
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0

    ### Bart system
    output_dir='feedback_sum'
    do_predict=True
    max_source_length=1024
    max_target_length=56
    data_dir="feedback"


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def main():
    config = Config()
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = BartSystem.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    # Some values from Config class needs to be copied to args to work.
    setattr(config, "num_train_epochs", args.num_train_epochs)
    setattr(config, "save_path", args.output_dir)
    setattr(args, "learning_rate", config.lr_F)

    # Create output directory.
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    setattr(config, "save_folder", os.path.join(config.save_path, timestamp))
    os.makedirs(os.path.join(config.save_folder, 'ckpts'))
    init_logger(config.save_folder)
    logger = logging.getLogger(__name__)

    model_F = BartSystem(args).to(config.device)
    # Don't use the trainer to fit the model
    args.do_train = False
    # trainer = generic_train(model_F, args)
    if args.output_dir:
        try:
            checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
            if checkpoints[-1]:
                BartSystem.load_from_checkpoint(checkpoints[-1])
                logger.info("Load checkpoint sucessfully!")
        except:
            logger.info("Failed to load checkpoint!")

    # train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    train_iters, dev_iters, test_iters = model_F.train_dataloader(), model_F.val_dataloader(), model_F.test_dataloader()
    model_D = Discriminator(config, model_F.tokenizer).to(config.device)

    logger.info(config.discriminator_method)
    # import pdb
    # pdb.set_trace()
    logger.info(model_D)

    train(config, model_F, model_D, train_iters, dev_iters, test_iters)
    

if __name__ == '__main__':
    main()
