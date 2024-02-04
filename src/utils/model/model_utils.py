# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    BitsAndBytesConfig
)
from transformers.deepspeed import HfDeepSpeedConfig

import torch.nn as nn


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False,
                    quantize=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    if rlhf_training:
        model = model_class.from_config(model_config)
    else:
        if not quantize:
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config)
        else:
            world_size = int(os.environ["WORLD_SIZE"])
            max_memory = f'{32000}MB'
            max_memory = {i: max_memory for i in range(world_size)}
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                quantization_config=bnb_config,)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        
    return model

