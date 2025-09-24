import gc
import streamlit as st
import computation_helper
import os
from datetime import datetime
import json
import io
import zipfile
import tempfile
from concurrent.futures import ProcessPoolExecutor

def parse_jsonl(file):
    # Use ProcessPoolExecutor for parallel parsing
    with ProcessPoolExecutor() as executor:
        return list(executor.map(json.loads, file))

def count_unique_prompts(json_lines):
    unique_sys_prompt = {}
    for json_line in json_lines:
        sys_prompt = json_line['conversations'][0]['value']
        unique_sys_prompt[sys_prompt] = unique_sys_prompt.get(sys_prompt, 0) + 1
    return unique_sys_prompt

# prevents the app from rerunning after downloading the processed zip file.
if 'zip_buffer' not in st.session_state:
    st.session_state['zip_buffer'] = None
    st.session_state['last_file_name'] = None


file = st.file_uploader("**Upload a JSONL file**", type=["jsonl"], accept_multiple_files=False)

if file:
    # Only process if new file or not already processed
    if st.session_state['last_file_name'] != file.name:
        try:
            json_lines = parse_jsonl(file)
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            st.stop()
        
        file_name = file.name

        # Flatten uploaded file from memory
        del file
        gc.collect()

        unique_sys_prompt = count_unique_prompts(json_lines)
        num_sys_prompts = len(unique_sys_prompt)
        st.write("Number of unique system prompts:", num_sys_prompts)

        st.write("Unique system prompts and their counts:")
        record = [
            {"prompt": prompt[:128] + ("..." if len(prompt) > 128 else ""), "count": count}
            for prompt, count in unique_sys_prompt.items()
        ]
        st.write(record)

        # File processing options
        percentage_lint = st.slider(label="**Top Percentage**", min_value=1, max_value=100, value=None)
        st.session_state['percentage_lint'] = percentage_lint
        include_csv = st.checkbox(label="**Include CSV files with the token counts**", key="include_metadata")
      
      
        # Temp folder to hold files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Classify system prompts and save to JSONL files
            token_counter = computation_helper.TextMatcher(temp_dir=temp_dir, num_sys_prompts=num_sys_prompts)
            sys_prompt_order = {prompt: i+1 for i, prompt in enumerate(unique_sys_prompt.keys())}
            token_counter.classify_prompts_and_token_counts(data=json_lines, prompts=sys_prompt_order)

            # Generate top percentile lint
            classified_csv_files = [file for file in os.listdir(f'{temp_dir}/classified_token_count') if file.endswith('.csv')]
            st.write(classified_csv_files)

            top_percentile_lint = computation_helper.TopPercentileLint(
                percentile=percentage_lint,
                classified_csv_files=classified_csv_files,
                temp_directory=temp_dir
            )
            top_percentile_lint.top_percentile_linter(jsonl_file=json_lines)

            # Flatten read lines from memory
            del json_lines
            gc.collect()

            # Create zip archive in memory from files in temp_dir
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for root, _, files in os.walk(f'{temp_dir}/classified_token_count/top_{percentage_lint}_percent'):
                    for file in files:
                        if not include_csv and file.endswith('.csv'):
                            continue  # Skip CSV files if include_csv is False
                        file_path = os.path.join(root, file)
                        with open(file_path, "rb") as f:
                            zip_file.writestr(file, f.read())
            zip_buffer.seek(0)

            st.write(f"**Download top **{percentage_lint}** lint per unique system prompt.**")

            # At the end, store the buffer and file name:
            st.session_state['zip_buffer'] = zip_buffer
            st.session_state['last_file_name'] = file_name
    else:
        zip_buffer = st.session_state['zip_buffer']

    
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Download button
    st.download_button(
        label="Download",
        data=st.session_state['zip_buffer'],
        file_name=f"top_{st.session_state['percentage_lint']}_percent_system_prompts_{now_str}.zip",
        mime="application/zip",
        disabled=(st.session_state['percentage_lint'] == 0)
    )

    # Flatten zip file from memory
    del zip_buffer
    gc.collect()