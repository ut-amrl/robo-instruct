from unsloth import FastLanguageModel
import torch 

name = "llama3_evol_ri_exp_lr3e-5_peft_constant_r128_a128"

for i in range(1300, 1400, 100):
    print(f"Processing checkpoint {i}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = f"/home/zichaohu/research/llm/robo-instruct/train/models/{name}/checkpoint-{i}",
        max_seq_length = 2048,
        dtype = torch.float16,
        load_in_4bit = False,
    )

    model.save_pretrained_merged(f"models/{name}_merged/checkpoint-{i}", tokenizer, save_method = "merged_16bit",)