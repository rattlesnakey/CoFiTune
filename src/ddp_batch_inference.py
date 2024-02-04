
# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from transformers import LlamaForCausalLM
import argparse, os, sys
import deepspeed
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.distributed as dist
from inference_utils import ddp_batch_generate, Mydataset
import json
from utils.utils import load_sub_weights
import subprocess
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


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
        # specify master port
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

def ddp_batch_inference(test_file_path:str, output_file_path:str):
    
    prompts = []

    if test_file_path.endswith(".json"):
        instructions = json.load(open(test_file_path, encoding='utf-8'))
    else:
        import jsonlines as jsl
        with jsl.open(test_file_path, 'r') as reader:
            instructions = list(reader)
    
    with torch.no_grad():
        for idx, item in enumerate(instructions):
            instruction = item.get('instruction', '') + '\n' + item.get('input', '')
            prompt = PROMPT_TEMPLATE.format_map({'instruction':instruction})
            prompts.append((idx, prompt))

        dataset = Mydataset(prompts=prompts)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=32, pin_memory=True)
        dataloader = accelerator.prepare_data_loader(data_loader=dataloader)

        results = ddp_batch_generate(
            tokenizer=tokenizer, 
            model=model,
            device=device,
            dataloader=dataloader, 
            do_sample=False, 
            repetition_penalty=1.2,
            temperature=0.0, 
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            accelerator=accelerator,
        )

        accelerator.wait_for_everyone()
        all_results = [None for _ in range(world_size)]
        dist.all_gather_object(all_results, results)

        if accelerator.is_main_process:
            final_all_results = dict()
            final_result_dicts = []

            for results in all_results:
                final_all_results.update(results)
    
            for idx, line in enumerate(instructions):
                result = final_all_results[str(idx)]
                answer = result.replace("<s>", "").replace("</s>", "").replace('<unk>',"").replace("[PAD]","").strip().split("### Response:")[1].strip()
                line['infer'] = answer
                line['index'] = idx

                final_result_dicts.append(line)
            

        if accelerator.is_main_process:
            with open(output_file_path, "w+", encoding='utf-8') as f:
                json.dump(final_result_dicts, f, ensure_ascii=False, indent=4)

            accelerator.print(output_file_path)
        
            




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--masterport", type=int, default=14444)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--tk_path", required=True, type=str)
    parser.add_argument("--test_path", default="data/jt_test_35_0728.jsonl",
                        type=str)
    parser.add_argument("--args_output_path", default="",
                        type=str)
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch size')
    parser.add_argument('--max_new_tokens', default=512, type=int, required=False, help='new tokens')

    
    args = parser.parse_args()
    setup_distributed(args.masterport)
    args.local_rank = int(os.environ["LOCAL_RANK"])
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        accelerator = Accelerator(kwargs_handlers=[kwargs], device_placement=False)

    args.global_rank = torch.distributed.get_rank()

    
    accelerator.print(args)
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    
    
    if args.local_rank <= 0:
        if '|||' in args.save_path:
            os.makedirs(os.path.dirname(args.save_path.split('|||')[0]), exist_ok=True)
        else:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    accelerator.print('loadding tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(args.tk_path, use_fast=False)
    
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
    

    accelerator.print(f'Using {device} ...')
    

    if args.args_output_path != 'None':
        args_output_path = os.path.join(args.args_output_path, 'args.json')

        with open(args_output_path, 'r') as f:
            accelerator.print(f'loading args from {args_output_path}...')
            model_args = json.load(f)

            if model_args['only_optimize_some_params'] != 'None' or model_args['only_optimize_some_blocks'] != 'None':
                accelerator.print(f"loading model from {model_args['model_name_or_path']} ...")
                model = LlamaForCausalLM.from_pretrained(model_args['model_name_or_path'], torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device) 
                sub_weight_dir = os.path.join(model_args['output_dir'], f"epoch_{model_args['num_train_epochs'] - 1}")
                sub_weight_path = os.path.join(sub_weight_dir, "adapter_model.bin")
                accelerator.print(f"loading sub weights from {sub_weight_path} ...")
                model = load_sub_weights(model, sub_weight_path)
                model.to(device)
            else:
                output_ckpt = os.path.join(model_args['output_dir'], f"epoch_{model_args['num_train_epochs'] - 1}")
                accelerator.print(f"loading model from {output_ckpt} ...")
                model = LlamaForCausalLM.from_pretrained(output_ckpt, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

    else:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
 

    model.eval()
    accelerator.print('Using Deepspeed accelerating ...')
    accelerator.print(f'Local rank:{args.local_rank} \t World_size:{world_size}')
    model.to(device)
    model = deepspeed.init_inference(model, mp_size=1, dtype=torch.float16, replace_with_kernel_inject=True)

    
    ddp_batch_inference(args.test_path, args.save_path)
    accelerator.print('Done')



        

    
    
    
