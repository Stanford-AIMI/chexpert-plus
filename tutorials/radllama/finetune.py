import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import Trainer, GPTQConfig
from transformers.integrations import deepspeed
from transformers.trainer import has_length
from transformers.trainer_pt_utils import LabelSmoother, LengthGroupedSampler

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="StanfordAIMI/RadLLaMA-7b")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"human": "<|user|>", "gpt": "<|assistant|>"}
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        system = [{"from": "system", "value": system_message if "system" not in source else source["system"]}]
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]
        assert len(source) >= 2
        input_id = tokenizer.apply_chat_template(system + source, return_tensors="pt")[0]
        target = torch.full(input_id.shape, -100, dtype=torch.int64)
        for i in range(1, len(source), 2):
            assert source[i]["from"] == "gpt"
            start_idx = len(
                tokenizer.apply_chat_template(system + source[:i], add_generation_prompt=True, return_tensors="pt")[0]
            )
            end_idx = len(
                tokenizer.apply_chat_template(system + source[:i + 1], return_tensors="pt")[0]
            )
            target[start_idx:end_idx] = input_id[start_idx:end_idx]
        input_id = input_id.tolist() + [tokenizer.pad_token_id] * (max_len - len(input_id))
        target = target.tolist() + [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids)
    targets = torch.tensor(targets)
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    @property
    def lengths(self):
        length_list = []
        for sample in self.raw_data:
            if "conv" in sample:
                text = " ".join([conv["value"] for conv in sample['conv'] if conv["from"] == "gpt"])
            else:
                text = sample["text"]
            num_img_tokens = 256 * text.count("<|img|>")
            length = len(text.split()) + num_img_tokens
            length_list.append(length)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        if "conv" in self.raw_data[i]:
            ret = preprocess([self.raw_data[i]["conv"]], self.tokenizer, self.max_len)
            ret = dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                attention_mask=ret["attention_mask"][0],
            )
        else:
            assert "text" in self.raw_data[i]
            tokenized = self.tokenizer(
                self.raw_data[i]["text"], max_length=self.max_len, padding="max_length", truncation=True,
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"][0]
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
            attention_mask = tokenized["attention_mask"][0]
            ret = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        self.cached_data_dict[i] = ret
        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def infer_model_class_and_trainable_parameters(training_args):
    model_config = transformers.AutoConfig
    model_class = transformers.AutoModelForCausalLM
    tokenizer_class = transformers.AutoTokenizer
    return model_config, model_class, tokenizer_class, training_args


class CustomTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        # Build the sampler.
        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    # Load model and tokenizer
    model_config, model_class, tokenizer_class, training_args = infer_model_class_and_trainable_parameters(
        training_args
    )

    config = model_config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        quantization_config=GPTQConfig(bits=4, disable_exllama=True)
        if training_args.use_lora and lora_args.q_lora else None,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True
    )
    tokenizer = tokenizer_class.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id

    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
