# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from ast import arg
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_trainable_param(model, args):
    if args.only_optimize_some_blocks != 'None':
        block_layers = args.only_optimize_some_blocks.split(',')
    else:
        block_layers = None

    if args.only_optimize_some_params != 'None':
        train_params = args.only_optimize_some_params.split(',')
    else:
        train_params = None
    

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)

    
    if train_params != None and block_layers != None:
        for name, parameter in model.named_parameters():
            for block_layer in block_layers:
                block_layer_name = f'layers.{block_layer}.'
                for train_param in train_params:
                    if block_layer_name in name and train_param in name:
                        parameter.requires_grad_(True)



    elif train_params == None and block_layers != None:
        for name, parameter in model.named_parameters():
            for block_layer in block_layers:
                block_layer_name = f'layers.{block_layer}.'
                if block_layer_name in name:
                    parameter.requires_grad_(True)

    
    elif train_params != None and block_layers == None:
        for name, parameter in model.named_parameters():
            for train_param in train_params:
                if train_param in name:
                    parameter.requires_grad_(True)
           
    elif train_params == None and block_layers == None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(True)

        
    

   

def count_learnable_params(model, args):
    learnable_param = 0
    frozen_param = 0
    if args.global_rank <= 0:
        for p in model.parameters():
            if p.requires_grad:
                try:
                    learnable_param+=p.ds_numel
                except Exception:
                    learnable_param+=p.numel()
            else:
                try:
                    frozen_param+=p.ds_numel
                except Exception:
                    frozen_param+=p.numel()

        
        total_param = frozen_param+learnable_param

        print_rank_0("Number of learnable parameters: {}".format(learnable_param), args.global_rank)
        print_rank_0("Number of frozen parameters: {}".format(frozen_param), args.global_rank)
        print_rank_0(f"Trainable%: {learnable_param/total_param}", args.global_rank)

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v 
    return output


def save_args(args, global_rank):
    args_save_path = os.path.join(args.args_output_path, 'args.json')
    if global_rank == 0:
        with open(args_save_path, 'w+') as f:
            json.dump(vars(args), f, indent=4)

def save_hf_format(model, tokenizer, args, sub_folder=""):
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    output_dir = os.path.join(args.output_dir, sub_folder)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)
    print_rank_0(output_dir)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     trainable_params=None, args=None,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    if trainable_params:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n
                                for nd in no_decay_name_list) and n in trainable_params)
                ],
                "weight_decay":
                weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (any(nd in n
                            for nd in no_decay_name_list) and n in trainable_params)
                ],
                "weight_decay":
                0.0,
            },
        ]
    
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n
                                for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay":
                weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay":
                0.0,
            },
        ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    print_rank_0(save_dir)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            model_to_save.save_pretrained(save_dir, max_shard_size="4GB")
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        for k, v in model_to_save.named_buffers():
            if global_rank == 0:
                output_state_dict[k] = v.cpu()
        if global_rank == 0:
            model_to_save.save_pretrained(save_dir, state_dict=output_state_dict, max_shard_size="4GB")
        del output_state_dict



def zero3_get_model_state_dict(model, global_rank, is_stage_3=False, save_embed_and_lm_head=False, candidate_keys=None):
    if not candidate_keys:
        all_params = set(k for k, _ in model.named_parameters())
        candidate_keys = set(k for k in all_params if "lora_" in k)

    if save_embed_and_lm_head:
        for k in all_params:
            if 'embed' in k or 'lm_head' in k:
                candidate_keys.add(k)
    if not is_stage_3:
        if global_rank ==0:
            state_dict = {
                k: v.cpu() for k, v in model.named_parameters()
                if k in candidate_keys
            }
            to_return = {k: v for k, v in state_dict.items()}
        else:
            to_return = None
    else:
        state_dict = {}
        for k, v in model.named_parameters():
            if k not in candidate_keys:
                continue
            if hasattr(v, 'ds_id') and k in candidate_keys:
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=is_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0:
                state_dict[k] = v_p
        if global_rank == 0:
            to_return = {k: v for k, v in state_dict.items()}
        else:
            to_return = None
    return to_return

def zero3_save_trainable_param(model, save_directory, global_rank, is_stage_3, save_embed_and_lm_head, trainable_params=None):
    model = model.module if hasattr(model, 'module') else model

    if not trainable_params:
        trainable_params = set()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                print_rank_0(f"*********** saving trainable param .... ************", global_rank)
                print_rank_0(f"module: {name}", global_rank)
                trainable_params.add(name)

    """Modified from PeftModel.save_pretrained"""
    WEIGHTS_NAME = "adapter_model.bin"

    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

    os.makedirs(save_directory, exist_ok=True)
    output_state_dict = zero3_get_model_state_dict(model, global_rank, is_stage_3=is_stage_3, save_embed_and_lm_head=save_embed_and_lm_head, candidate_keys=trainable_params)
    if global_rank == 0:
        
        torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME))

        print(f"Trainable weights saved to {save_directory}")


def load_sub_weights(model, sub_weights_path):
    state_dict = torch.load(sub_weights_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model