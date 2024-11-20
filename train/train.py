
from transformers import TrainingArguments, logging
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
import os 
import torch 
from unsloth import FastLanguageModel
import hydra 
from omegaconf import DictConfig, OmegaConf
os.environ["TOKENIZERS_PARALLELISM"] = "true"

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    training_args = TrainingArguments(**cfg.training_args)
    os.environ["WANDB_PROJECT"] = cfg.misc_args.wandb_project_name  # name your W&B project
    if cfg.misc_args.verbose:
        logging.set_verbosity_debug()
    if cfg.data_args.load_from_disk:
        train_dataset = load_from_disk(cfg.data_args.train_data_name_or_path)["train"]
    else:
        train_dataset = load_dataset(cfg.data_args.train_data_name_or_path, split="train")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg.model_args.model_name_or_path,
        max_seq_length = cfg.misc_args.seq_length,
        load_in_4bit=False
    )


    dataset_text_field = cfg.data_args.text_column
    def inspect_inputs(inputs):
        for idx in range(len(inputs)):
            input_ids = inputs[idx]['input_ids']
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"Decoded Text: {decoded_text}")
            print(tokens)
            print(f"Tokens: {len(tokens)}")
        input()
    data_collator=lambda data: inspect_inputs(data)
    data_collator =None

    tokenizer.pad_token = tokenizer.eos_token
    # Must add EOS_TOKEN
    def add_eos_to_batch(examples):
        examples[cfg.data_args.text_column] = [
            text + tokenizer.eos_token for text in examples[cfg.data_args.text_column]
        ]
        return examples
    train_dataset = train_dataset.map(add_eos_to_batch, batched=True)

    trainer = SFTTrainer(
        model=model,
        args = training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        packing = cfg.misc_args.packing,
        max_seq_length = cfg.misc_args.seq_length,
        dataset_text_field=dataset_text_field,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_model()

    
if __name__ == "__main__":
    main()