import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
# from trainer import LisaTrainer
import transformers
import trl
import wandb
import torch
from peft import LoraConfig, get_peft_model, PeftModel
wandb.init(mode="disabled")


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    rho: float = field(default=0)
    opt_method: str = field(default="sft")
    previous_lora: str = field(default="")
    reasoning_step: str = field(default=2)
    alignment_step: str = field(default=50)
    previous_lora: str = field(default="")
    system_evaluate: str = field(default="False")
    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "cache_dir":"cache" , "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        kwargs = { "use_cache": False, "torch_dtype": torch.bfloat16}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name,**kwargs)
        
    # if len(config.previous_lora)>0:
    #     # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #     model = PeftModel.from_pretrained(
    #     model,
    #     config.previous_lora,
    #     is_trainable=True
    #     )
    # else:
    #     lora_config = LoraConfig(
    #                     # r=500,
    #                     r=32,
    #                     lora_alpha=8,
    #                     target_modules=["q_proj", "k_proj", "v_proj"],
    #                     lora_dropout=0,
    #                     bias="none",
    #                     task_type="CAUSAL_LM",
    #                     )
    #             # initialize the model with the LoRA framework
    #     # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #     model = get_peft_model(model, lora_config)   
     
    model = model.to(torch.bfloat16)
    
    
    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    
    # model.push_to_hub("TianshengHuang/s1k")
    # tokenizer.push_to_hub("TianshengHuang/s1k")
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    else:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    
    
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    args.rho=config.rho
    print(model)
    print(dataset['train'])
    if config.opt_method=="sft":
        trainer = trl.SFTTrainer(
            model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            data_collator=collator
        )
    # elif config.opt_method == "lisa":
    #     args.reasoning_step = config.reasoning_step
    #     args.alignment_step = config.alignment_step
    #     trainer = LisaTrainer(
    #         model,
    #         train_dataset=dataset['train'],
    #         eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
    #         args=args,
    #         data_collator=collator
    #     )
    #     reasoning_dataset = load_dataset("TianshengHuang/s1k_small")["train"]
    #     trainer.init(reasoning_dataset)
        # print("hihi")
    else:
        import sys
        sys.exit()
        
        
    num_train_samples = len(dataset['train'])
    num_train_epochs = args.num_train_epochs
    train_batch_size = args.per_device_train_batch_size*8
    gradient_accumulation_steps = args.gradient_accumulation_steps
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    total_steps = num_train_epochs * (num_train_samples // effective_batch_size)
    print(total_steps)
    from transformers import TrainerCallback
    class GPUTimeCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic = 0
            self.record_time = 0
        
        def on_step_begin(self, args, state, control, **kwargs):
            state.start_event = torch.cuda.Event(enable_timing=True)
            state.end_event = torch.cuda.Event(enable_timing=True)
            state.start_event.record()
    

        def on_step_end(self, args, state, control, **kwargs):
            state.end_event.record()
            torch.cuda.synchronize()
            step_time = state.start_event.elapsed_time(state.end_event)
            self.average_statistic =  (self.average_statistic* self.record_time +step_time) / (self.record_time+1)  
            self.record_time +=1
            if self.record_time%50==0:
                # print(f"Step {state.global_step}: {self.average_statistic*self.record_time / 1000:.2f} seconds (GPU time)")
                print("Estimated total time {} (h)".format(self.average_statistic*total_steps/ 1000/3600))
                
    
    class GPUMemoryCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic_memory = 0
            self.record_time_memory = 0
        
        def on_step_begin(self, args, state, control, **kwargs):
            state.start_memory = torch.cuda.memory_reserved()
            # print(self.record_time_memory)
            
        def on_step_end(self, args, state, control, **kwargs):
            state.end_memory = torch.cuda.memory_reserved()
            self.average_statistic_memory =  (self.average_statistic_memory* self.record_time_memory +state.end_memory ) / (self.record_time_memory+1)  
            self.record_time_memory +=1
            if self.record_time_memory%50==0:
                print(f"Step {state.global_step}: {self.average_statistic_memory / (1024 ** 3):.2f} GB GPU memory used")
                
    
    if config.system_evaluate =="True":
        trainer.add_callback(GPUTimeCallback())
        trainer.add_callback(GPUMemoryCallback())
    
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # tokenizer.save_pretrained(args.output_dir)
    # trainer.model.push_to_hub("TianshengHuang/"+args.output_dir.split("/")[-1])
    # tokenizer.push_to_hub("TianshengHuang/"+args.output_dir.split("/")[-1])
    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":

    train()
