import numpy as np
import pandas as pd
import time
from datetime import datetime
import pytz

import json
from openai import OpenAI
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError, Timeout, RequestException
# from openai.error import APIConnectionError, APIError
from json.decoder import JSONDecodeError
import os

# Load the API key from a file
with open('/Users/admin/Documents/Github/openai_secretkey.txt', 'r', encoding='utf-8') as file:
    api_key_text = file.read().strip()

# Instantiate the OpenAI client with the API key
client = OpenAI(api_key=api_key_text)

# Print the current time in EST
print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))

# List of CSV files to process
csv_files = ['SpacingOut_congruent.csv', 'SpacingOut_neutral.csv', 'SpacingOut_incongruent.csv']
temperature_settings = [1]
model_settings = ['gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106','gpt-4-1106-preview','gpt-4-0613']
iterations_N = 20

for model_setting in model_settings:
    for csv_file in csv_files:
        for temperature_setting in temperature_settings:
            for iteration in range(iterations_N):
                print('Iteration:', iteration)

                # Extract condition from CSV file name
                condition = csv_file.split('_')[1].split('.')[0]

                # Path to the CSV file containing the math problems
                csv_file_path = './' + csv_file

                # Output file paths with condition in name
                output_json_outfile_path = './output_temp' + str(temperature_setting) + '_' + condition + '_' + model_setting + '_' + str(iteration) + '.json'
                output_csv_outfile_path = './output_temp' + str(temperature_setting) + '_' + condition + '_' + model_setting + '_' + str(iteration) + '.csv'
                output_pivot_df1_outfile_path = './pivot_df1_temp' + str(temperature_setting) + '_' + condition + '_' + model_setting + '_' + str(iteration) + '.csv'
                output_pivot_df2_outfile_path = './pivot_df2_temp' + str(temperature_setting) + '_' + condition + '_' + model_setting + '_' + str(iteration) + '.csv'
                html_outfile_path = './data_table_temp' + str(temperature_setting) + '_' + condition + '_' + model_setting + '_' + str(iteration) + '.html'

                # Loading the CSV file into a pandas DataFrame
                df = pd.read_csv(csv_file_path)

                # List of 10 user prompts extracted from random_df
                user_prompts = df['Problem'].tolist()

                # model = 'gpt-4-1106-preview'
                # model = 'gpt-3.5-turbo'

                # Grid of parameters
                parameters = [
                #{'model': 'gpt-3.5-turbo', 'temperature': 0.8, 'max_tokens': 300, 'seed': 123},
                    #'model': 'gpt-3.5-turbo', 'temperature': 0.5, 'max_tokens': 300, 'seed': 123},
                    # {'model': 'gpt-4-1106-preview', 'temperature': 0.8, 'max_tokens': 500},
                    {'model': model_setting, 'temperature': temperature_setting, 'max_tokens': 400}
                    # ... (other combinations of parameters)
                ]






                # def chat_with_gpt(messages, model="gpt-3.5-turbo", temperature=0.8, max_tokens=300, seed=128, retries=3):
                def chat_with_gpt(messages, model="gpt-3.5-turbo", temperature=0.8, max_tokens=400, seed = 123, retries=300):
                    """
                    Function to interact with OpenAI's GPT model in chat mode.
                    Includes a retry mechanism for handling various exceptions.
                    """
                    attempt = 0
                    while attempt < retries:
                        try:
                            response = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                seed = 123 + iteration,
                                timeout = 30
                            )
                            # Accessing the response content
                           # Accessing the response content
                            response_content = response.choices[0].message.content.strip()
                            return response_content
                        except ReadTimeout as e:
                            if attempt < retries - 1:
                                attempt += 1
                                print(f"Read timeout occurred. Retrying {attempt}/{retries} after 1 minutes...")
                                print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                time.sleep(60)  # Delay for 2 minutes
                            else:
                                print("Maximum retries reached after timeout. Exiting.")
                                print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                raise
                        except (ConnectionError, HTTPError, Timeout, RequestException, JSONDecodeError) as e:
                            if attempt < retries - 1:
                                attempt += 1
                                print(f"Error occurred: {e}. Retrying {attempt}/{retries} after 1 minutes...")
                                print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                time.sleep(60)
                                continue
                            else:
                                print("Maximum retries reached. Raising exception...")
                                print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                raise
                        except Exception as e:
                            # Catch-all for any other exceptions
                            if "server is overloaded or not ready yet" in str(e):
                                if attempt < retries - 1:
                                    attempt += 1
                                    print(f"Server is overloaded. Retrying {attempt}/{retries} after 1 minutes...")
                                    print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                    time.sleep(60)  # Delay for 1 minutes
                                else:
                                    print("Maximum retries reached after server overload. Exiting.")
                                    print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                    raise
                            else:
                                print(f"Unexpected error occurred: {e}.")
                                if attempt < retries - 1:
                                    attempt += 1
                                    print(f"Retrying {attempt}/{retries} after 1 minute...")
                                    print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                    time.sleep(60)  # Delay for 1 minutes
                                else:
                                    print(f"Maximum retries reached after unexpected error: {e}. Exiting...")
                                    print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                    raise


                # List of 4 system prompts
                # (Important: MINIMAL!) is very reductionist.
                system_prompts = [
                    """You are a student in grade 6 who knows nothing about order of operations (PEMDAS) in Mathematics.
                    How would you answer if given this Mathematics question on order of operations?
                    Stick to how the answer may be given using this given persona and nothing else.
                    Explain your solution (concise!) to this given Mathematics problem (using the above persona given), and put the final answer in the last line in the form "The answer is __." 
                    Replace __ with the final answer.
                    """,
                    """You are a student in grade 6 who knows nothing about order of operations (PEMDAS) in Mathematics.
                    Answer the Mathematics question using this given persona and nothing else.
                    Explain your solution (concise!) to this given Mathematics problem (using the above persona given), and put the final answer in the last line in the form "The answer is __." 
                    Replace __ with the final answer.
                    """,
                    """You are a student in elementary school who knows nothing about order of operations (PEMDAS) in Mathematics.
                    How would you answer if given this Mathematics question on order of operations?
                    Stick to how the answer may be given using this given persona and nothing else.
                    Explain your solution (concise!) to this given Mathematics problem (using the above persona given), and put the final answer in the last line in the form "The answer is __." 
                    Replace __ with the final answer.
                    """,
                    """You are a student in elementary school who knows nothing about order of operations in Mathematics.
                    Answer the Mathematics question using this given persona and nothing else.
                    Explain your solution (concise!) to this given Mathematics problem (using the above persona given), and put the final answer in the last line in the form "The answer is __." 
                    Replace __ with the final answer.
                    """,
                ]



                # Initialize an empty list to hold all responses
                all_responses = []

                # Iterating over system and user prompts
                for i, system_prompt in enumerate(system_prompts, start=1):
                    for j, user_prompt in enumerate(user_prompts, start=1):
                        conversation_history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                        for param in parameters:
                            gpt_response = chat_with_gpt(conversation_history, model=param['model'], temperature=param['temperature'], max_tokens=param['max_tokens'], retries = 200, seed=123+iteration)
                            conversation_history.append({"role": "assistant", "content": "GPT Response: " + gpt_response})
                            
                            # Create a unique identifier for each conversation
                            file_name = f"systemPrompt{i}_userPrompt{j}_model{param['model']}_temp{param['temperature']}_maxtoken{param['max_tokens']}"
                            
                            # Store the conversation history and parameters in the list
                            all_responses.append({
                                "file_name": file_name,
                                "model": param['model'],
                                "temperature": param['temperature'],
                                "max_tokens": param['max_tokens'],
                                "conversation_history": conversation_history
                            })
                            print(f"Conversation {file_name} completed")

                # Write all the conversation histories and parameters to the file once
                with open(output_json_outfile_path, 'w', encoding='utf-8') as f:
                    json.dump(all_responses, f)

                print("All conversations saved to file")


                # Function to normalize and flatten the JSON data
                def normalize_json_data(all_responses):
                    normalized_data = []

                    # Iterating through each element in the JSON list
                    for entry in all_responses:
                        file_name = entry.get('file_name', '')
                        model = entry.get('model', '')
                        temperature = entry.get('temperature', '')
                        max_tokens = entry.get('max_tokens', '')

                        # Iterating through each conversation history item
                        for conversation in entry.get('conversation_history', []):
                            # Extracting conversation details
                            role = conversation.get('role', '')
                            content = conversation.get('content', '')

                            # Constructing a row with all needed information
                            row = {
                                'file_name': file_name,
                                'model': model,
                                'temperature': temperature,
                                'max_tokens': max_tokens,
                                'role': role,
                                'content': content
                            }
                            normalized_data.append(row)

                    return pd.DataFrame(normalized_data)

                # Creating a DataFrame from the normalized JSON data
                normalized_df = normalize_json_data(all_responses)

                # Display the first few rows of the DataFrame
                normalized_df.head()

                # Save the DataFrame as a CSV file
                normalized_df.to_csv(output_csv_outfile_path, index=False)
                print("Json saved to file output.csv")

                # Function to extract system prompt number from the file name
                def extract_system_prompt(file_name):
                    # Extracting the system prompt part from the file name
                    parts = file_name.split("_")
                    for part in parts:
                        if part.startswith("systemPrompt"):
                            return part
                    return None

                # Function to extract user prompt number from the file name
                def extract_user_prompt(file_name):
                    # Extracting the system prompt part from the file name
                    parts = file_name.split("_")
                    for part in parts:
                        if part.startswith("userPrompt"):
                            return part
                    return None

                # Adding a new column 'system_prompt' to the DataFrame
                normalized_df['system_prompt'] = normalized_df['file_name'].apply(extract_system_prompt)
                normalized_df['user_prompt_N'] = normalized_df['file_name'].apply(extract_user_prompt)

                # Creating a pivot table to restructure the DataFrame
                pivot_df = normalized_df.pivot_table(
                    index=['model', 'temperature', 'max_tokens', 'system_prompt', 'user_prompt_N'],
                    columns='role',
                    values='content',
                    aggfunc='first'
                ).reset_index()

                # Renaming columns for clarity
                pivot_df = pivot_df.rename(columns={'user': 'user_prompt', 'assistant': 'systemPrompt'})

                # Reordering columns to match the requested format
                ordered_columns = ['model', 'temperature', 'max_tokens', 'system_prompt', 'user_prompt', 'systemPrompt']
                pivot_df = pivot_df[ordered_columns]

                pivot_df.to_csv(output_pivot_df1_outfile_path, index=False)

                print("Pivot table saved to file pivot_df1")

                # pivot_df = pd.read_csv('./pivot_df1.csv')

                # Creating separate columns for each system prompt
                new_pivot_df = pivot_df.pivot_table(
                    index=['model', 'temperature', 'max_tokens', 'user_prompt'],
                    columns='system_prompt',
                    values='systemPrompt',
                    aggfunc='first'
                ).reset_index()

                new_pivot_df = new_pivot_df[['model', 'temperature', 'user_prompt', 
                                            'systemPrompt1', 'systemPrompt2', 'systemPrompt3', 'systemPrompt4']]
                new_pivot_df.to_csv(output_pivot_df2_outfile_path, index=False)
                print("Pivot table saved to file pivot_df2")
                new_pivot_df = pd.read_csv(output_pivot_df2_outfile_path)

                import pandas as pd
                from jinja2 import Template

                # Replace the "×" symbol with HTML entity "&times;" in the DataFrame
                new_pivot_df.replace('×', '&times;', inplace=True, regex=True)


                def format_code_block(cell):
                    # This function looks for text within triple backticks and formats it
                    if '```' in cell:
                        parts = cell.split('```')
                        for i in range(1, len(parts), 2):
                            # Preserve newlines and apply a monospaced font within triple backticks
                            parts[i] = f"<pre style='font-family: monospace;'>{parts[i]}</pre>"
                        return ''.join(parts)
                    return cell

                def handle_newlines(cell):
                    # Modify this function to call format_code_block
                    if isinstance(cell, str):
                        cell = format_code_block(cell)  # Format code blocks if present
                        return cell.replace('\n', '<br>')
                    return cell

                # Apply the newline handling function to other columns
                for column in new_pivot_df.columns:
                    new_pivot_df[column] = new_pivot_df[column].apply(handle_newlines)

                # Convert DataFrame to HTML without escaping (to keep LaTeX format)
                html_table = new_pivot_df.to_html(index=False, escape=False)

                # HTML template with MathJax support for LaTeX
                html_template = """
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Data Table</title>
                    <script type="text/javascript" async
                    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
                    </script>
                    <style>
                        table {
                            border-collapse: collapse;
                            width: 100%;
                            table-layout: fixed;
                        }
                        th, td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                            word-wrap: break-word;
                        }
                        th {
                            background-color: #f2f2f2;
                        }
                    </style>
                </head>
                <body>
                    <h1>Data Overview</h1>
                    {{ table_html|safe }}
                </body>
                </html>
                """

                template = Template(html_template)
                rendered_html = template.render(table_html=html_table)

                # Saving the HTML file
                with open(html_outfile_path, 'w', encoding='utf-8') as file:
                    file.write(rendered_html)

                print("HTML file saved as 'data_table.html' in UTF-8 encoding")


                # Import the necessary modules
                from datetime import datetime
                import pytz

                # Get the current time with timezone information
                utc_now = datetime.now(pytz.utc)

                # Create a timezone object for Eastern Standard Time (EST)
                est_timezone = pytz.timezone('US/Eastern')

                # Convert the UTC time to EST
                est_time = utc_now.astimezone(est_timezone)

                # Print the current time in EST
                print("Current time in EST:", est_time.strftime('%Y-%m-%d %H:%M:%S %Z'))

print("Done with OpenAI!")
print("##############################")
print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))

import os
import pandas as pd
import re
from jinja2 import Template

def combine_csv_files(outfile, outhtml):
    """
    Combine all CSV files in the current directory that start with 'pivot_df2_' into a single DataFrame.

    Parameters:
    outfile (str): The filename for the output CSV file.
    outhtml (str): The filename for the output HTML file.

    Returns:
    pd.DataFrame: The combined DataFrame with an added 'spacing' column.
    """
    all_dfs = []  # List to store all DataFrames
    folder_path = os.getcwd()  # Get the current working directory

    for filename in os.listdir(folder_path):
        if filename.startswith('pivot_df2_') and filename.endswith('.csv'):
            # Extract the '*' value from the filename
            match = re.search(r'pivot_df2_temp(?:0|1)_(.*?)_', filename)
            parts = filename.replace('.csv', '').split('_')
            iterX = parts[-1]
            if match:
                spacing_value = match.group(1)

                # Load the CSV file into a DataFrame
                df = pd.read_csv(os.path.join(folder_path, filename))

                # Insert the 'iteration' column as the first column
                df.insert(0, 'iteration', iterX)

                # Insert the 'spacing' column as the first column
                df.insert(0, 'spacing', spacing_value)

                # Add the DataFrame to the list
                all_dfs.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(outfile, index=False)

    new_pivot_df = pd.read_csv(outfile)



    # Replace the "×" symbol with HTML entity "&times;" in the DataFrame
    new_pivot_df.replace('×', '&times;', inplace=True, regex=True)


    def format_code_block(cell):
        # This function looks for text within triple backticks and formats it
        if '```' in cell:
            parts = cell.split('```')
            for i in range(1, len(parts), 2):
                # Preserve newlines and apply a monospaced font within triple backticks
                parts[i] = f"<pre style='font-family: monospace;'>{parts[i]}</pre>"
            return ''.join(parts)
        return cell

    def handle_newlines(cell):
        # Modify this function to call format_code_block
        if isinstance(cell, str):
            cell = format_code_block(cell)  # Format code blocks if present
            return cell.replace('\n', '<br>')
        return cell

    # Apply the newline handling function to other columns
    for column in new_pivot_df.columns:
        new_pivot_df[column] = new_pivot_df[column].apply(handle_newlines)

    # Convert DataFrame to HTML without escaping (to keep LaTeX format)
    html_table = new_pivot_df.to_html(index=False, escape=False)

    # HTML template with MathJax support for LaTeX
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Data Table</title>
        <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
        </script>
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                table-layout: fixed;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                word-wrap: break-word;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h1>Data Overview</h1>
        {{ table_html|safe }}
    </body>
    </html>
    """

    template = Template(html_template)
    rendered_html = template.render(table_html=html_table)

    # Saving the HTML file
    with open(outhtml, 'w', encoding='utf-8') as file:
        file.write(rendered_html)

    print("HTML file saved as 'data_table.html' in UTF-8 encoding")


    # Import the necessary modules
    from datetime import datetime
    import pytz

    # Get the current time with timezone information
    utc_now = datetime.now(pytz.utc)

    # Create a timezone object for Eastern Standard Time (EST)
    est_timezone = pytz.timezone('US/Eastern')

    # Convert the UTC time to EST
    est_time = utc_now.astimezone(est_timezone)

    # Print the current time in EST
    print("Current time in EST:", est_time.strftime('%Y-%m-%d %H:%M:%S %Z'))

    return combined_df

# Use the function to combine files and generate output
combine_csv_files('pivot_df2_combined.csv', 'data_table_combined.html')


import pandas as pd
import re

# Load the CSV files into pandas dataframes

df_combined = pd.read_csv('pivot_df2_combined.csv')
SpacingOut_answer = pd.read_csv('SpacingOut_answer.csv')


# Task (a): Creating a new variable "user_prompt_trimmed" from "user_prompt" by removing all spaces
df_combined['user_prompt_trimmed'] = df_combined['user_prompt'].str.replace(' ', '')

# Task (b): Deleting column "user_prompt"
df_combined.drop('user_prompt', axis=1, inplace=True)

# Task (c): Restructuring df_combined
# Melt the dataframe to transform systemPrompt columns into rows
melted_df_combined = pd.melt(df_combined, id_vars=['spacing', 'iteration', 'model', 'temperature', 'user_prompt_trimmed'],
                             value_vars=['systemPrompt1', 'systemPrompt2', 'systemPrompt3',
                                         'systemPrompt4'],
                             var_name='prompt_number', value_name='systemPrompt')

# Replacing 'systemPromptX' with just the number X in 'prompt_number' column
melted_df_combined['prompt_number'] = melted_df_combined['prompt_number'].str.replace('systemPrompt', '').astype(int)

# Pivoting the table to have different spacings as columns
restructured_df_combined = melted_df_combined.pivot_table(index=['model', 'temperature','user_prompt_trimmed', 'prompt_number', 'iteration'],
                                                          columns='spacing', values='systemPrompt', aggfunc='first').reset_index()


# Task (a): Removing "GPT Response: " from each cell in the specified columns
columns_to_edit = ['congruent', 'incongruent', 'neutral']
for col in columns_to_edit:
    restructured_df_combined[col] = restructured_df_combined[col].str.replace('GPT Response: ', '')

# Task (b): Extracting the value XX from the last sentence "answer is XX." in each cell
# and creating new columns for each answer type
# for col in columns_to_edit:
    # Extracting the answer from the last sentence in each cell
    # restructured_df_combined[f'{col}_ans'] = restructured_df_combined[col].str.extract(r'answer is (\d+).$')

# Adjusting the answer extraction logic to use the last few characters " XX." where XX is the value
# for col in columns_to_edit:
    # Extracting the answer from the last few characters in each cell
    # restructured_df_combined[f'{col}_ans'] = restructured_df_combined[col].str.extract(r' (\d+)\.$')

# Adjusting the answer extraction logic with an if-else approach
# First, try extracting using " XX." logic, including negative numbers
# If that fails (i.e., results in null), then try extracting from "answer is XX."

# Define the function to extract answer from quotes
def extract_answer_from_quotes(text):
    if pd.isna(text):
        return None
    parts = text.split('"')
    for part in parts:
        if "The answer is" in part:
            match = re.search(r'The answer is ([-]?\d+)', part)
            if match:
                return match.group(1)
    return None


for col in columns_to_edit:
    # First attempt: Extracting the answer from the last few characters in each cell, including negative numbers
    restructured_df_combined[f'{col}_ans'] = restructured_df_combined[col].str.extract(r' ([-]?\d+)[.! ]$')

    # Second attempt: If the first extraction is null, try extracting from "answer is XX."
    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].str.extract(r'answer is ([-]?\d+)[.! ]$')
    
    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].str.extract(r'answer is ([-]?\d+)[.! ]')
    
    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].str.extract(r'answer is ([-]?\d+).$')

    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].str.extract(r'answer is ([-]?\d+)')
    
    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].str.extract(r'"The answer is ([-]?\d+)."')

    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].str.extract(r'The answer is ([-]?\d+)')

    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].str.extract(r"'The answer is ([-]?\d+).'")
    
    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].str.extract(r"The answer is (\d+)")

    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].str.extract(r"The answer is ['\"]?(\d+)['\"]?")

    # New logic: Extract answer from quotes if still null
    restructured_df_combined.loc[restructured_df_combined[f'{col}_ans'].isnull(), f'{col}_ans'] = \
        restructured_df_combined[col].apply(extract_answer_from_quotes)


# Performing the left join as specified in part (b)
restructured_df_combined = pd.merge(restructured_df_combined, SpacingOut_answer, left_on='user_prompt_trimmed', right_on='Problem', how='left')

# Creating new variables as specified in part (c)
# For each type (congruent, incongruent, and neutral), compare the answers and assign 1 if they match, 0 if they don't, and null if either is null
for ans_type in ['congruent', 'incongruent', 'neutral']:
    ans_col = f'{ans_type}_ans'
    correct_col = f'{ans_type}_ansCorrect'
    restructured_df_combined[correct_col] = restructured_df_combined.apply(lambda row: 1 if row[ans_col] == str(row['Answer']) else (0 if pd.notnull(row[ans_col]) else None), axis=1)



# Save file to csv
restructured_df_combined.to_csv('restructured_df_combined_spacing.csv', index=False)




