#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaTokenizer,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


from utils.data.distributed_dataset import DatasetSpliter, LocalSampler
from utils.utils import print_rank_0, save_args, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.utils import zero3_save_trainable_param, count_learnable_params, set_trainable_param, save_args, str2bool
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model

#! Soft-masking
from softmask_utils import compute_impt_vector, load_overall_softmask, apply_softmask_to_grad
from softmask_llama import SoftMaskLlamaForCausalLM

import subprocess
import json


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_config_path',
                        type=str,
                        default='data_config.json',
                        help='Path to the config of data ...')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="warmup_ratio")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--gradient_checkpointing',
                        type=str2bool,
                        default=False,
                        help='Enable HF gradient checkpointing for model.')

    
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help=
        "Path to tokenizer",
        required=True,
    )

    parser.add_argument(
        "--args_output_path",
        type=str,
        default='outputs',
        required=True,
    )
    
    # deepspeed features
    parser.add_argument('--offload',
                        type=str2bool,
                        default=False,
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    parser.add_argument(
        "--masterport",
        type=int,
        default=13333,
        help=
        "node masterport",
    )

    ## only optimize some blocks
    parser.add_argument('--only_optimize_some_blocks',
                        type=str,
                        default='None',
                        help='only optimize some blocks, e.g. 1,2,3,4 blocks')
    
    ## only optimize some params
    parser.add_argument('--only_optimize_some_params',
                        type=str,
                        default='None',
                        help='only optimize some params, e.g. ffn, attention, input | output projection')

    
    ## softmask 
    parser.add_argument('--compute_impt_vector',
                        type=str2bool,
                        default=False,
                        help='To get important vector')
    
    parser.add_argument('--train_with_softmask',
                        type=str2bool,
                        default=False,
                        help='in SFT stage, use softmask or not')
    
    parser.add_argument('--softmask_path',
                        type=str,
                        default="None",
                        help='the softmask saved dir')

    parser.add_argument('--apply_softmask',
                        type=str,
                        default="up_projection,down_projection",
                        help='the modules you want to apply the softmask')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def setup_distributed(port=None):
    """
    Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            os.environ["MASTER_PORT"] = "13333"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    )


def converter(data, tokenizer, max_seq_len):
    instruction = data['instruction'] + '\n' + data['input']
    instruction = PROMPT_TEMPLATE.format_map({'instruction':instruction})
    sentence = instruction + data['output']

    token = tokenizer(sentence,
                                max_length=max_seq_len,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")
    token["input_ids"] = token["input_ids"].squeeze(0)
    token["attention_mask"] = token["attention_mask"].squeeze(0)

    pad_id = tokenizer.pad_token_id
    pad_list = token['input_ids'] == pad_id
    num_of_pad = pad_list.count_nonzero()

    labels = token["input_ids"].clone()

    labels[:num_of_pad] = -100

    return {
        "input_ids": token["input_ids"],
        "attention_mask": token["attention_mask"],
        "labels": labels
    }

def main():
    setup_distributed(args.masterport)
    args.local_rank = int(os.environ["LOCAL_RANK"])
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

   
    set_random_seed(args.seed)
    torch.distributed.barrier()

    


    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                              fast_tokenizer=True,
                                              add_eos_token=True)
    if (len(tokenizer))!=49954:
        raise ValueError(f"The vocab size of the tokenizer must be 49954, but found {len(tokenizer)}.\n"
                         "Please use Chinese Alpaca tokenizer!")
    if tokenizer.pad_token is None:
        print(f"Adding pad token {DEFAULT_PAD_TOKEN}")
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    
    if tokenizer.bos_token is None:
        print(f"Adding bos token {DEFAULT_BOS_TOKEN}")
        tokenizer.add_special_tokens(dict(bos_token=DEFAULT_BOS_TOKEN))
    
    if tokenizer.eos_token is None:
        print(f"Adding eos token {DEFAULT_EOS_TOKEN}")
        tokenizer.add_special_tokens(dict(eos_token=DEFAULT_EOS_TOKEN))
    
    if tokenizer.unk_token is None:
        print(f"Adding unk token {DEFAULT_UNK_TOKEN}")
        tokenizer.add_special_tokens(dict(unk_token=DEFAULT_UNK_TOKEN))

    tokenizer.padding_side = "left" 

    if args.compute_impt_vector:
        ModelClass = SoftMaskLlamaForCausalLM
        print_rank_0('using softmask version Llama Model', args.global_rank)
    else:
        ModelClass = AutoModelForCausalLM
   
    model = create_hf_model(ModelClass,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config)

    
    print_rank_0(args, args.global_rank)
    

    if args.compute_impt_vector:
        print_rank_0(f"***** computing softmask .. *****", args.global_rank)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(False)
        #! zero3 have to include the trainable param for optimizer
        for name, parameter in model.named_parameters():
            if 'layers.0.' in name and 'q_proj' in name:
                parameter.requires_grad_(True)
        
        print_rank_0("***** trainable module name  *****", args.global_rank)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                print_rank_0(f"module: {name}", args.global_rank)
                
        count_learnable_params(model, args)
        
        if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
                model, args.weight_decay)
        
        optimizer = AdamOptimizer(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95))
    
        model, optimizer, _, _ = deepspeed.initialize(
                                        model=model,
                                        optimizer=optimizer,
                                        args=args,
                                        config=ds_config,
                                        dist_init_required=True
                                    )
        if args.gradient_checkpointing:
            print_rank_0("***** Using Gradient checkpointing .. *****", args.global_rank)
            model.gradient_checkpointing_enable()

        compute_impt_vector(model=model, data_loader=train_dataloader, args=args, device=device, optimizer=optimizer)
        print_rank_0('computing importance vector done !', args.global_rank)
        return

    
    if args.only_optimize_some_params != "None" or args.only_optimize_some_blocks != "None":
        set_trainable_param(model, args)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        count_learnable_params(model, args)
        print_rank_0("***** trainable module name  *****", args.global_rank)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                print_rank_0(f"module: {name}", args.global_rank)

        




    world_size = torch.distributed.get_world_size()

    spliter = DatasetSpliter(args.data_config_path, args.local_rank, args.global_rank, world_size, args.seed, tokenizer, converter, args.max_seq_len)
    train_dataset = spliter.train_dataset
    eval_dataset = spliter.eval_dataset
    train_sampler = LocalSampler(train_dataset, args.global_rank, args.seed)
    eval_sampler = LocalSampler(eval_dataset, args.global_rank, args.seed)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size,
                                  num_workers=0)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size,
                                 num_workers=0)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    
    if args.only_optimize_some_params != "None" or args.only_optimize_some_blocks != "None":
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay)
      
    
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
                model, args.weight_decay)


    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95))


    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.num_train_epochs * num_update_steps_per_epoch * args.warmup_ratio),
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

   
    
    
    print_rank_0("***** Deepspeed DDP initializing LLM .. *****", args.global_rank)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    
    if args.gradient_checkpointing:
        print_rank_0("***** Using Gradient checkpointing .. *****", args.global_rank)
        model.gradient_checkpointing_enable()


    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(f'train_dataloader len {len(train_dataloader)}', args.global_rank)
    print_rank_0(f'eval_dataloader len {len(eval_dataloader)}', args.global_rank)


    if args.train_with_softmask:
        softmask = load_overall_softmask(args, device=device)
    else:
        softmask = None

    training_loss = []
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)

            if args.train_with_softmask:
                apply_softmask_to_grad(model, softmask, args)

            model.step()
            print_rank_0(
                f"*** training loss: {loss.item()} *****", args.global_rank
            )
            training_loss.append(loss.item())
            
        model.tput_timer.update_epoch_count()

        if args.output_dir is not None:
            print_rank_0('saving the model ...', args.global_rank)
            save_directory = os.path.join(args.output_dir, f"epoch_{epoch}")
            save_args(args, args.global_rank)
            
            if args.global_rank <= 0:
                os.makedirs(save_directory, exist_ok=True)
                training_loss_path = os.path.join(args.args_output_path, 'training_loss.json')
                with open(training_loss_path, 'w+') as f:
                    print_rank_0('saving training loss log ..')
                    json.dump(training_loss, f)

            if args.only_optimize_some_blocks != "None" or args.only_optimize_some_params != "None":
                zero3_save_trainable_param(model, save_directory, args.global_rank, args.zero_stage == 3, save_embed_and_lm_head=False)
                
            else:
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args, sub_folder=f"epoch_{epoch}")
                save_zero_three_model(model,
                                        args.global_rank,
                                        save_directory,
                                        zero_stage=args.zero_stage)


if __name__ == "__main__":
    args = parse_args()
    main()