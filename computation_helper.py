import re
import os
import string
import json
import tiktoken
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


class TokenCounter:
    """Counts the number of tokens in a string.
    Uses the tiktoken library to encode the string based on the specified model/encoding.
    """
    
    def __init__(self, encoding_name='o200k_base'):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        tokens = self.encoding.encode(string)
        return (len(tokens))

class TextMatcher:
    def __init__(self, temp_dir, num_sys_prompts):
        self.files_directory = f"{temp_dir}" or 'classified/extracted_jsonl_and_csv_files'
        self.num_sys_prompts = num_sys_prompts or 9
        # os.makedirs(self.files_directory, exist_ok=True)

        self.output_jsonl_files = [f'{temp_dir}/prompt_type_{num}.jsonl' for num in range(1, self.num_sys_prompts + 1)]

        os.makedirs(f'{self.files_directory}/classified_token_count', exist_ok=True)
        self.output_csv_files = [f'{self.files_directory}/classified_token_count/prompt_type_{num}-token_counts.csv' for num in range(1, self.num_sys_prompts + 1)]

    @staticmethod
    def normalize_text(text):
        # Lowercase, remove punctuation, collapse whitespace
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def word_overlap_percentage(self, str1, str2):
        words1 = set(self.normalize_text(str1).split())
        words2 = set(self.normalize_text(str2).split())

        if not words1 or not words2:
            return 0.0
        
        shared = words1 & words2
        total = words1 | words2
        return round(len(shared) / len(total) * 100, 2)
    
    def classified_files_generator(self):
        # Create empty files or clear existing files
        for file_name in self.output_jsonl_files:
            with open(file_name, 'w', encoding='utf-8', errors='ignore') as f:
                pass
        
        for file_name in self.output_csv_files:
            with open(file_name, 'w', encoding='utf-8', errors='ignore') as f:
                f.write("line_num,system,human,gpt,total_token_count\n")

    def _process_item(self, args):
        i, item, prompts = args
        prompt_identifier = prompts[item['conversations'][0]['value']] - 1
        jsonl_file_path = self.output_jsonl_files[prompt_identifier]
        csv_file_path = self.output_csv_files[prompt_identifier]

        token_counter = TokenCounter()
        system_tokens = token_counter.num_tokens_from_string(item['conversations'][0]['value'])
        human_tokens = token_counter.num_tokens_from_string(item['conversations'][1]['value'])
        gpt_tokens = token_counter.num_tokens_from_string(item['conversations'][2]['value'])
        total_tokens = system_tokens + human_tokens + gpt_tokens

        return (
            prompt_identifier,
            json.dumps(item) + '\n',
            f"{i},{system_tokens},{human_tokens},{gpt_tokens},{total_tokens}\n"
        )

    def classify_prompts_and_token_counts(self, data, prompts):
        self.classified_files_generator()
        results = [[] for _ in range(self.num_sys_prompts)]
        csv_results = [[] for _ in range(self.num_sys_prompts)]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._process_item, (i, item, prompts)) for i, item in enumerate(data)]
            for future in as_completed(futures):
                prompt_identifier, jsonl_line, csv_line = future.result()
                results[prompt_identifier].append(jsonl_line)
                csv_results[prompt_identifier].append(csv_line)

        # Write results to files
        for idx, file_name in enumerate(self.output_jsonl_files):
            with open(file_name, 'a', encoding='utf-8', errors='ignore') as sys_prompt_file:
                sys_prompt_file.writelines(results[idx])
        for idx, file_name in enumerate(self.output_csv_files):
            with open(file_name, 'a', encoding='utf-8', errors='ignore') as csv_data:
                csv_data.writelines(csv_results[idx])

class TopPercentileLint:
    """Reads classified token count CSV files and calculates the top percentile  lint.  
    Then stores the corresponding rows in a new CSV file, and the corresponding JSONL data as well."""

    def __init__(self, classified_csv_files, temp_directory, percentile):
        self.classified_csv_files = classified_csv_files
        self.classified_csv_directory = f'{temp_directory}/classified_token_count'
        self.directory = f'{temp_directory}/classified_token_count/top_{percentile}_percent'
        self.percentile = percentile
        os.makedirs(self.directory, exist_ok=True)
        self.output_files = [f'{self.directory}/{pf[:-4]}_top_{self.percentile}' for pf in self.classified_csv_files]

    def top_percentile_files_generator(self):
        # Create empty files or clear existing files
        for fname in self.output_files:
            with open(f'{fname}.jsonl', 'w', encoding='utf-8', errors='ignore') as f:
                pass

    def top_percentile_linter(self, jsonl_file):
        csv_files = self.classified_csv_files
        for csv_file in csv_files:
            df = pd.read_csv(f'{self.classified_csv_directory}/{csv_file}')
            # Extract rows in the top percentile based on total_token_count
            percentile_i = df['total_token_count'].quantile((100 - self.percentile) / 100)
            filtered = df[df['total_token_count'] >= percentile_i]

            file_prefix = csv_file[:-4]
            # Save the filtered rows to a new CSV file
            filtered.to_csv(f'{self.directory}/{file_prefix}_top_{self.percentile}.csv', index=False)

        # Call the method to extract JSONL data
        self.extract_top_percentile_jsonl_lines(jsonl_file)


    def _extract_jsonl_lines_for_file(self, args):
        file, jsonl_file = args
        output_lines = []
        csv = pd.read_csv(f'{file}.csv')
        for _, row in csv.iterrows():
            output_lines.append(json.dumps(jsonl_file[row['line_num']]) + '\n')
        return file, output_lines

    def extract_top_percentile_jsonl_lines(self, jsonl_file):
        self.top_percentile_files_generator()
        args_list = [(file, jsonl_file) for file in self.output_files]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._extract_jsonl_lines_for_file, args) for args in args_list]
            for future in as_completed(futures):
                file, output_lines = future.result()
                with open(f'{file}.jsonl', 'a', encoding='utf-8', errors='ignore') as top_percentile_jsonl:
                    top_percentile_jsonl.writelines(output_lines)



if __name__ == "__main__":
    # Read lines from the JSONL file
    with open('merged_sharegpt_training_data.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]

    # # Files containing different prompts to compare against
    # prompt_files = [
    #     'prompts/another_sys_prompt_type.txt', # type_1
    #     'prompts/constraints_prompts.txt', # type_2
    #     'prompts/final_antwort_prompt.txt', # type_3
    #     'prompts/planning_prompts.txt', # type_4
    #     'prompts/sub_system_1.txt', # type_5
    #     'prompts/sub_system_2.txt' # type_6
    # ]

    # matcher = TextMatcher(temp_dir='extracted_jsonl_and_csv_files')
    # matcher.classify_prompts_and_token_counts(data)

    # Files containing classified token counts
    classified_csv_files = [
            "classified_token_count/another_sys_prompt_type.csv",
            "classified_token_count/constraints_prompts.csv",
            "classified_token_count/final_antwort_prompt.csv",
            "classified_token_count/planning_prompts.csv",
            "classified_token_count/sub_system_1.csv",
            "classified_token_count/sub_system_2.csv"
        ]

    linter = TopPercentileLint(classified_csv_files)
    linter.top_percentile_linter()