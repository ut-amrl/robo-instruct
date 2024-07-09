from robo_instruct.robosim.robosim_tracer import rejection_sampling_simulation

import argparse
import os
import pandas as pd
import numpy as np 
from joblib import Parallel, delayed

os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
def eval_program(args):
    data = pd.read_csv(args.input_name)
    columns = data.columns.tolist()
    has_gen_count = True
    if "gen_count" not in columns:
        has_gen_count = False
        columns.append("gen_count")

    def sub_eval(df_rows, idx):
        print("start idx", idx)
        gen_count = 0
        for index, row in df_rows.iterrows():
            gen_count += 1
            program = row["program"]
            if type(program) == type(np.nan):
                continue
            (trace_elements, status) = rejection_sampling_simulation(
                program=program, 
                simulation_timeout=2, 
                resampling_count=100, 
                allowed_timeout_count=2,
                VERBOSE=False
            )
            if status == True:
                return index, gen_count
            
        return -1, -1
    max_rejection_sampling_count = args.max_rejection_sampling_count
    eval_result = Parallel(n_jobs=-2)(delayed(sub_eval)(data.iloc[i:i+max_rejection_sampling_count], i) 
                                      for i in range(0, len(data), max_rejection_sampling_count)) # all but one cpus
    # combine data
    print("start combining data")
    result = []
    for (data_idx, gen_count) in eval_result:
        if gen_count < 0:
            continue 
        column_data = []
        for column in columns:
            if "gen_count" in column and not has_gen_count:
                tmp_data = gen_count
            else:
                tmp_data = data.iloc[data_idx][column]
            column_data.append(tmp_data)
        result.append(column_data)
    # apply dedup
    result_df = pd.DataFrame(result, columns=columns)
    result_df_dedup = result_df.drop_duplicates(subset=['prompt'])
    result_df_dedup = result_df_dedup.drop_duplicates(subset=['program'])
    result_df_dedup.to_csv(args.save_name, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_name", type=str, default="data/si_instruction_program_pairs.csv")
    parser.add_argument("-s", "--save_name", type=str, default="data/robosim_pairs.csv")
    parser.add_argument("--max_rejection_sampling_count", type=int, default=3, help="total repeat")
    args = parser.parse_args()
    eval_program(args)
