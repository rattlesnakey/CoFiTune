# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class Mydataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
        

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]
    


def ddp_batch_generate(
    tokenizer=None,
    model=None,
    device=None,
    dataloader=None, 
    do_sample=False,
    repetition_penalty=1.2,
    temperature=0.0,
    num_beams=1,
    max_new_tokens=512,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    accelerator=None,
    use_cache=True,
    **kwargs
):
    
    results = dict()
    
    for data_tuple in tqdm(dataloader, desc='Inferencing ...', total=len(dataloader), disable=accelerator.process_index):
        idxes, data = data_tuple
        idxes = idxes.tolist()
        data = list(data)
        
        inputs = tokenizer(data, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generation_output = model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=temperature,
                    use_cache=use_cache, 
                    pad_token_id=pad_token_id,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    return_dict_in_generate=True,
                )
            

        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s)
        accelerator.wait_for_everyone()

        
        for idx, output in zip(idxes, outputs):
            results.update({f'{idx}': f"{output}"})
            
    return results

