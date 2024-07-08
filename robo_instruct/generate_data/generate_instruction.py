from robo_instruct.misc.llm_generation_utils import post_process_llama3_instruction
from Levenshtein import distance

import os
import pandas as pd
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
  
def gather_prompts_self_instruct_vllm(llm, s_param, tokenizer, args, generate_n=20):
    existing_prompts = pd.read_csv(args.prompt_path, header=None)
    if os.path.exists(args.save_name):
        new_prompts = pd.read_csv(args.save_name, header=None)
    else:
        new_prompts = pd.DataFrame()
    if len(new_prompts) > 30:
        df1 = new_prompts.sample(n=2).values.tolist()
        df2 = existing_prompts.sample(n=4).values.tolist()
        df = df1 + df2
        print("using Self Instruct")
    else:
        df = existing_prompts.sample(n=6).values.tolist()
    
    system_inst = """
You are a helpful assistant. Here is a robot that has the following capabilities:
- def get_current_location() -> str:
- def get_all_rooms() -> list[str]:
- def is_in_room(object : str) -> bool:
- def go_to(location : str) -> None:
- def ask(person : str, question : str, options: list[str]) -> str:
- def say(message : str) -> None:
- def pick(obj: str) -> None:
- def place(obj: str) -> None:
"""
    prompt = [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[0][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[1][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[2][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[3][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[4][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[5][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
    ]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

    outputs = llm.generate([prompt] * generate_n, s_param)
    result = []
    for output in outputs:
        instruction = output.outputs[0].text
        instruction = post_process_llama3_instruction(instruction)
        result.append(instruction)
    return result

def deduplicate_instructions(args, df):
    print("Initial Size: ", len(df))
    prompts = list(df["prompt"])
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
    duplicated_idx = set()
    encoded_prompts = []
    result = []
    for p in prompts:
        encoded = tokenizer.encode(p)
        encoded_prompts.append(encoded)

    for i in range(len(encoded_prompts)):
        for j in range(i+1, len(encoded_prompts)):
            dist = distance(encoded_prompts[i], encoded_prompts[j])
            edit_sim = 1 - dist / max(len(encoded_prompts[i]), len(encoded_prompts[j]))
            if edit_sim > 0.6:
                duplicated_idx.add(j)
            result.append([i, j, dist, edit_sim])
    df.drop(list(duplicated_idx), inplace=True)
    print("Final Size: ", len(df))
    df.to_csv(args.save_name, index=False, header=["prompt"])

def main_vllm_llama3_inst_8b(args):
    llama3_llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = SamplingParams(
        temperature=0.999, 
        top_p=0.95,
        max_tokens=1024,
        stop_token_ids=stop_token_ids)
    final_result = []
    for i in range(5):
        print("generating batch: ", i)
        result = gather_prompts_self_instruct_vllm(llama3_llm, sampling_params, tokenizer, args, generate_n=10)
        final_result.extend(result)
    
    df = pd.DataFrame({"prompt": final_result})
    deduplicate_instructions(args, df)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt_path", type=str, default="seed_instructions.csv")
    parser.add_argument("-s", "--save_name", type=str, default="data/si_instructions.csv")
    args = parser.parse_args()
    main_vllm_llama3_inst_8b(args)
