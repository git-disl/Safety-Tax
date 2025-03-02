from trl import SFTTrainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn

from transformers import Trainer
from transformers import logging
import transformers
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import (
    is_sagemaker_mp_enabled
)

from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.gemma.modeling_gemma import GemmaAttention
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import copy

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

def get_leaf_modules_with_grad(module):

    module_list= []
    for name, module in module.named_modules():
        if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention) or isinstance(module, GemmaAttention) or isinstance(module, Qwen2Attention)or isinstance(module, Gemma2Attention):
            module_list+= [module]

    return module_list

class LisaTrainer(SFTTrainer):

    
    def get_reasoning_dataloader(self,reasoning_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        # Tokenize the dataset before passing to the collator
        
        # def tokenize_function(example):
        #     # print("kkkk")
        #     return self.tokenizer(
        #         example["text"],  # Assuming "text" contains the reasoning example
        #         padding="max_length",
        #         truncation=True,
        #         max_length=self.args.max_seq_length,
        #         return_tensors="pt",
        #     )

        # # Apply tokenization
        # # print("fuck1")
        # tokenized_dataset = reasoning_dataset.map(tokenize_function, batched=True, num_proc=1,keep_in_memory=True)
        # # print("fuck")
        # tokenized_dataset = tokenized_dataset.remove_columns(
        #     ["solution", "question", "cot_type", "source_type", "metadata", "cot", "thinking_trajectories", "attempt", "text"]
        # )

        # # Convert dataset to PyTorch tensors
        # tokenized_dataset.set_format("torch")
        processing_class = transformers.AutoTokenizer.from_pretrained(self.model.config._name_or_path, use_fast=True)
        
        processing_class.pad_token = processing_class.eos_token
        
        # formatting_func = get_formatting_func_from_dataset(reasoning_dataset, processing_class)
        #     # if a template is detected, we don't need to add special tokens again
        # if formatting_func is not None:
            # self.args.dataset_kwargs["add_special_tokens"] = False
            
        tokenized_dataset = self._prepare_dataset(
                    reasoning_dataset,
                    processing_class,
                    self.args.packing,
                    self.args.dataset_text_field,
                    self.args.max_seq_length,
                    None,
                    self.args.num_of_sequences,
                    self.args.chars_per_token,
                    remove_unused_columns=self.args.remove_unused_columns if self.args is not None else True,
                    **self.args.dataset_kwargs,
                )
        # print(tokenized_dataset)
        # print(self.train_dataset)
        tokenized_dataset = self._remove_unused_columns(tokenized_dataset, description="reasoning")
        data_collator = self.data_collator
  
        sampler = RandomSampler(tokenized_dataset)

        
        
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(tokenized_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(tokenized_dataset, **dataloader_params))
    
    
    def init(self,  reasoning_dataset):
        # first reasoning then alignment, in the end the weigths are naturally switch to alignment weights
        # self.reasoning_dataset=reasoning_dataset
        
        self.status = "reasoning"
                # self.gamma[name]= torch.zeros_like(param)
        self.clock = 0
        self.steps = 0
        
        self.reasoning_dataloader = self.get_reasoning_dataloader(reasoning_dataset)
        self.data_iter = iter(self.reasoning_dataloader)
        # self.gamma = {}
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         self.gamma[name] = 0
        
        
    # def end_training(self):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             if self.status == "alignment":
    #                 self.alignment_weights[name] = param.data.detach().clone()
    #             else:
    #                 self.finetune_weights[name] = param.data.detach().clone()
        
        
        
        
    
    def switch_model(self):
        sum_drift =0
        if self.status == "reasoning":
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.alignment_weights[name] = param.data.detach().clone()
                    # print(self.reasoning_weights.keys())
                    sum_drift += torch.norm(self.reasoning_weights[name] - self.alignment_weights[name])**2
            print("alignment to consensus{}".format(sum_drift))
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.reasoning_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.reasoning_weights[name] - self.alignment_weights[name])**2
            print("reasoning drift to consensus{}".format(sum_drift))
        
        
        
    def sample_from_reasoning(self):
        # Get a  batch
        try:
            batch = next(self.data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.data_iter = iter(self.reasoning_dataloader)
            batch = next(self.data_iter)
        return batch
    
    
    def check_mode(self, inputs):
        if self.status == "alignment":
            if self.clock% (self.args.alignment_step )  ==  0 and self.steps!=0 and self.args.reasoning_step!=0:
                self.status ="reasoning"
                # self.switch_model()
                # print("swith from alignment to reasoning {}".format(self.steps))
                # reasoning need another input
                inputs = self.sample_from_reasoning()
                self.clock=0
        else:
            if  self.clock% (  self.args.reasoning_step  )  ==  0 and self.steps!=0 and self.args.alignment_step!=0:
                self.status ="alignment"
                # self.switch_model()
                self.clock=0
            else:
                # reasoning need another input
                inputs = self.sample_from_reasoning()
                
        # if self.steps% (self.args.alignment_step + self.args.reasoning_step):
        #     # time to update gamma
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             self.gamma[name] = self.gamma[name] + self.args.rho* (self.reasoning_weights[name]-self.alignment_weights[name])
        return inputs
            
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        # if self.steps==0:
        #     self.alignment_weights ={}
        #     self.reasoning_weights ={}
        #     # self.gamma ={}
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             self.alignment_weights[name] = param.data.detach().clone()
        #             self.reasoning_weights[name] = param.data.detach().clone()
        # may change input due to mode change
        inputs = self.check_mode(inputs)
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                
            # if self.status =="alignment":
            #     # print("alignment_loss_prev: {}".format(loss.item()))
            #     for name, param in model.named_parameters():
            #         if param.requires_grad and self.args.rho>0:
            #             loss += (torch.sum(self.gamma[name] *  param )) +self.args.rho/2* torch.norm( param- self.alignment_weights[name])**2
            # else:
            #     for name, param in model.named_parameters():
            #         if param.requires_grad and self.args.rho>0:
            #             loss += (- torch.sum(self.gamma[name] *  param )) + self.args.rho/2* torch.norm( param- self.reasoning_weights[name])**2
         
                # print("finetune_loss: {}".format(loss.item()))
        
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 
        
        
        loss = step()    
        self.steps+=1
        self.clock+=1
        return loss.detach() / self.args.gradient_accumulation_steps