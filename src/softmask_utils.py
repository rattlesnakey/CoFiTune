import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import os
from torch import nn
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

import torch.nn.functional as F



def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v 
    return output

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def gather_by_mean(head_impt):
    head_impt_list = [torch.zeros_like(head_impt) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=head_impt_list,
                    tensor=head_impt.contiguous()) 
    head_impt_list = torch.stack(head_impt_list)
    head_impt = torch.mean(head_impt_list, dim=0)
    return head_impt

def impt_norm(impt):
    tanh = torch.nn.Tanh()
    for layer in range(impt.size(0)):
        impt[layer] = (impt[layer] - impt[layer].mean()) / impt[
            layer].std()  
    impt = tanh(impt).abs()

    return impt


def initial_impt(config, dtype, device):
    n_hidden_layer, n_attention_heads = config.num_hidden_layers, config.num_attention_heads
    intermediate_impt = torch.zeros(n_hidden_layer, config.intermediate_size, dtype=torch.float32).to(device)
    intermediate_mask = torch.ones(n_hidden_layer, config.intermediate_size, dtype=dtype).to(device)
    intermediate_mask.requires_grad_(requires_grad=True)

    output_impt = torch.zeros(n_hidden_layer, config.hidden_size, dtype=torch.float32).to(device)
    output_mask = torch.ones(n_hidden_layer, config.hidden_size, dtype=dtype).to(device)
    output_mask.requires_grad_(requires_grad=True)

    head_impt = torch.zeros(n_hidden_layer, n_attention_heads, dtype=torch.float32).to(device)
    head_mask = torch.ones(n_hidden_layer, n_attention_heads, dtype=dtype).to(device)
    head_mask.requires_grad_(requires_grad=True)
    tot_tokens = 0.0

    return  head_impt, intermediate_impt, output_impt,head_mask, intermediate_mask, output_mask, tot_tokens

def compute_impt_vector(model, data_loader, args, device, optimizer):
    
    head_impt, intermediate_impt, output_impt, \
    head_mask, intermediate_mask, output_mask, tot_tokens = initial_impt(config=model.module.config, dtype=model.module.dtype, device=device)

    softmask = {
        'input_projection': intermediate_mask,
        'output_projection': output_mask,
        'attention': head_mask,
    }
    model.train()
    for step, inputs in enumerate(tqdm(data_loader, desc="Computing softmask ..", disable=args.global_rank)):
        inputs = to_device(inputs, device)
        outputs1 = model(
                    **inputs, use_cache=False,
                    softmask=softmask
                )

        outputs2 = model(
                    **inputs, use_cache=False,
                    softmask=softmask
                )
        T = 0.1 
        loss = F.kl_div(F.log_softmax(outputs1.logits, dim=-1), F.softmax(outputs2.logits, dim=-1), reduction='batchmean') * T
        print_rank_0(f'loss:{loss.item()}', args.global_rank)
        loss.backward()

        head_impt += head_mask.grad.detach()
        intermediate_impt += intermediate_mask.grad.detach()
        output_impt += output_mask.grad.detach()
        tot_tokens += inputs["attention_mask"].float().detach().sum().data
        optimizer.zero_grad(set_to_none=True)

    head_impt /= tot_tokens

    intermediate_impt /= tot_tokens
    output_impt /= tot_tokens

    dist.barrier()


    head_impt = gather_by_mean(head_impt)
    intermediate_impt = gather_by_mean(intermediate_impt)
    output_impt = gather_by_mean(output_impt)

    if args.global_rank <= 0:
        softmask_dir = args.output_dir
        os.makedirs(softmask_dir, exist_ok=True)
        np.save(os.path.join(softmask_dir, "head_impt.npy"), head_impt.detach().cpu().numpy())
        np.save(os.path.join(softmask_dir, "intermediate_impt.npy"),intermediate_impt.detach().cpu().numpy())
        np.save(os.path.join(softmask_dir, "output_impt.npy"), output_impt.detach().cpu().numpy())
        
        print_rank_0(f"Softmasks saved to {softmask_dir} ..", args.global_rank)

    return head_impt, intermediate_impt, output_impt

def load_overall_softmask(args, device):
    if args.softmask_path != 'None':
        print_rank_0(f'loading soft-mask from {args.softmask_path} ...', args.global_rank)
        all_parts = {'input_projection':0, 'output_projection':0, 'attention':0}
        inp_proj_soft_mask, out_proj_soft_mask, attention_soft_mask = None, None, None
        
        if args.apply_softmask != 'None':
            apply_parts = args.apply_softmask.split(',')
            for a in all_parts:
                if a not in apply_parts:
                    print_rank_0(f'not using {a} softmask !!', args.global_rank)
                else:
                    print_rank_0(f'using {a} softmask !!', args.global_rank)
                    all_parts[a] = 1
        else:
            soft_mask = None
            return soft_mask
        
        attention_soft_mask_list = []; inp_proj_soft_mask_list = []; out_proj_soft_mask_list = []
        softmasks = os.listdir(args.softmask_path)

        for cur_softmask_name in softmasks:
            cur_softmask_path = os.path.join(args.softmask_path, cur_softmask_name)
            print_rank_0(f'loading softmask {cur_softmask_name} from {cur_softmask_path} ...')
            
            if all_parts['input_projection']:
                inp_proj_soft_mask = torch.Tensor(np.load(os.path.join(cur_softmask_path, "intermediate_impt.npy"))).to(device)
                inp_proj_soft_mask = impt_norm(inp_proj_soft_mask)
                inp_proj_soft_mask_list.append(inp_proj_soft_mask)

            if all_parts['output_projection']:
                out_proj_soft_mask = torch.Tensor(np.load(os.path.join(cur_softmask_path, "output_impt.npy"))).to(device)
                out_proj_soft_mask = impt_norm(out_proj_soft_mask)
                out_proj_soft_mask_list.append(out_proj_soft_mask)
            
            if all_parts['attention']:
                attention_soft_mask = torch.Tensor(np.load(os.path.join(cur_softmask_path, "head_impt.npy"))).to(device)
                attention_soft_mask = impt_norm(attention_soft_mask)
                attention_soft_mask_list.append(attention_soft_mask)

        if len(attention_soft_mask_list) > 0:
            attention_soft_masks = torch.stack(attention_soft_mask_list)
            attention_soft_mask, _ = attention_soft_masks.max(0)
        
        if len(inp_proj_soft_mask_list) > 0:
            inp_proj_soft_masks = torch.stack(inp_proj_soft_mask_list)
            inp_proj_soft_mask, _ = inp_proj_soft_masks.max(0)
        
        if len(out_proj_soft_mask_list) > 0:
            out_proj_soft_masks = torch.stack(out_proj_soft_mask_list)
            out_proj_soft_mask, _ = out_proj_soft_masks.max(0)

        soft_mask = {
            'input_projection':inp_proj_soft_mask, 
            'output_projection':out_proj_soft_mask,
            'attention':attention_soft_mask
        }

   
    else:
        soft_mask = None
    return soft_mask


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def apply_softmask_to_grad(model, softmask, args):
    config = model.module.config
    n_layers, n_heads = config.num_hidden_layers, config.num_attention_heads
    head_size = int(config.hidden_size / config.num_attention_heads)
    zero_stage_3 = (args.zero_stage == 3)

    if softmask != None:
        for layer in range(n_layers):
            if softmask['attention'] != None:
                cur_q_param, cur_k_param, cur_v_param = model.module.model.layers[layer].self_attn.q_proj.weight, model.module.model.layers[layer].self_attn.k_proj.weight, model.module.model.layers[layer].self_attn.v_proj.weight
                temp_attn_params = [cur_q_param, cur_k_param, cur_v_param]
                params_to_fetch = _z3_params_to_fetch(temp_attn_params) if zero_stage_3 else []
                should_gather_param = len(params_to_fetch) > 0
                with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=should_gather_param):
                    cur_q_grad = deepspeed.utils.safe_get_full_grad(cur_q_param)
                    
                    if cur_q_grad != None:
                        head_impt = softmask['attention'][layer].unsqueeze(-1).repeat((1, head_size))
                        head_impt = head_impt.flatten()
                        head_mask = 1 - head_impt
                        
                        cur_q_param._z3_optimizer.set_mask_to_grad_for_param(cur_q_param, head_mask, args)
                        cur_k_param._z3_optimizer.set_mask_to_grad_for_param(cur_k_param, head_mask, args)
                        cur_v_param._z3_optimizer.set_mask_to_grad_for_param(cur_v_param, head_mask, args)
                        print_rank_0(f'applying softmask to the gradient of layer {layer} attention ...', args.global_rank)

            if softmask['input_projection'] != None:
                cur_param = model.module.model.layers[layer].mlp.up_proj.weight
                params_to_fetch = _z3_params_to_fetch([cur_param]) if zero_stage_3 else []
                should_gather_param = len(params_to_fetch) > 0
                with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=should_gather_param):
                    cur_input_grad = deepspeed.utils.safe_get_full_grad(cur_param)
                    if cur_input_grad != None:
                        intermediate_mask = (1 - softmask['input_projection'][layer])
                        cur_param._z3_optimizer.set_mask_to_grad_for_param(cur_param, intermediate_mask, args)
                        print_rank_0(f'applying softmask to the gradient of layer {layer} input_projection ...', args.global_rank)
        
            if softmask['output_projection'] != None:
                cur_param = model.module.model.layers[layer].mlp.down_proj.weight
                params_to_fetch = _z3_params_to_fetch([cur_param]) if zero_stage_3 else []
                should_gather_param = len(params_to_fetch) > 0
                with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=should_gather_param):
                    cur_output_grad = deepspeed.utils.safe_get_full_grad(cur_param)
                    if cur_output_grad != None:
                        output_mask = (1 - softmask['output_projection'][layer])
                        cur_param._z3_optimizer.set_mask_to_grad_for_param(cur_param, output_mask, args)
                        print_rank_0(f'applying softmask to the gradient of layer {layer} output_projection ...', args.global_rank)
                
        print_rank_0(f'applying softmask to the gradient of module {args.apply_softmask} ...', args.global_rank)
    else:
        print_rank_0(f'Do not apply softmask to the gradient !!! ', args.global_rank)

    