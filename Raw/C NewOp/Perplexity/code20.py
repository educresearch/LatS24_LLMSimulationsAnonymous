import numpy as np
import pandas as pd
import time
from datetime import datetime
import pytz

import json
import requests
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError, Timeout, RequestException
# from openai.error import APIConnectionError, APIError
from json.decoder import JSONDecodeError

# Print the current time in EST
print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))

# List of CSV files to process
csv_files = ['SpacingOut_congruent.csv', 'SpacingOut_neutral.csv', 'SpacingOut_incongruent.csv']
temperature_settings = [1]
model_settings = ['mistral-7b-instruct', 'mixtral-8x7b-instruct', 'llama-2-70b-chat']
iterations_N = 20

# Define the Perplexity API URL and Headers
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Function to read the token from a file
def read_token_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# Specify the path to your token file
token_file_path = '/Users/admin/Documents/Github/perplexity_secretkey.txt'  # Update this path to the actual file location

# Read the token from the file
api_token = read_token_from_file(token_file_path)

# Set up the headers with the token read from the file
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {api_token}"
}


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
                    {'model': model_setting, 'temperature': temperature_setting, 'max_tokens': 700}
                    # ... (other combinations of parameters)
                ]


                def chat_with_perplexity(messages, model="mistral-7b-instruct", temperature=0.8, max_tokens=700, retries=300):
                    """
                    Includes a retry mechanism for handling various exceptions.
                    """
                    attempt = 0
                    while attempt < retries:
                        try:
                            payload = {
                               "model": model,
                               "messages": messages,
                               "max_tokens": max_tokens,
                               "temperature": temperature
                            }
                            response = requests.post(
                                PERPLEXITY_API_URL, 
                                json=payload, 
                                headers=HEADERS)
                            response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
                            return response.json()      # Adjust based on the response structure of Perplexity API
                        except ReadTimeout as e:
                            if attempt < retries - 1:
                                attempt += 1
                                print(f"Read timeout occurred. Retrying {attempt}/{retries} after 1 minute...")
                                print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                time.sleep(60)  # Delay for 1 minutes
                            else:
                                print("Maximum retries reached after timeout. Exiting.")
                                print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                raise
                        except (ConnectionError, HTTPError, Timeout, RequestException,
                                # APIConnectionError, 
                                JSONDecodeError) as e:
                            if attempt < retries - 1:
                                attempt += 1
                                print(f"Error occurred: {e}. Retrying {attempt}/{retries}...")
                                print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                time.sleep(60) # Delay for 1 minutes
                                continue
                            else:
                                print("Maximum retries reached. Raising exception...")
                                print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                raise
                        # except APIError as e:
                        #     if attempt < retries - 1:
                        #         attempt += 1
                        #         print(f"APIError occurred: {e}. Retrying {attempt}/{retries} after 5 minutes...")
                        #         print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                        #         time.sleep(120)  # Delay for 2 minutes
                        #         continue
                        #     else:
                        #         print("Maximum retries reached after APIError. Exiting.")
                        #         print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                        #         raise
                        except Exception as e:
                            # Catch-all for any other exceptions
                            if "server is overloaded or not ready yet" in str(e):
                                if attempt < retries - 1:
                                    attempt += 1
                                    print(f"Server is overloaded. Retrying {attempt}/{retries} after 5 minutes...")
                                    print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                    time.sleep(120)  # Delay for 2 minutes
                                else:
                                    print("Maximum retries reached after server overload. Exiting.")
                                    print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                    raise
                            else:
                                print(f"Unexpected error occurred: {e}. Exiting...")
                                # Print the current time in EST
                                print("Current time in EST:", datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S %Z'))
                                raise


                # List of 4 system prompts
                # (Important: MINIMAL!) is very reductionist.
                system_prompts = [
                    """There is a new development in mathematics convention involving operations and order of operations. 
                    Two new operations were defined:
                    1. (C) Concatenation operation: Define a @ b as the concatenation of the digits of |a| and |b|. This operation simply joins the numbers at a digit level.
                    Key Examples of Concatenation (C):
                    (a) If a = 12 and b = 56, then a @ b = 12 @ 56 = 1256.
                    (b) If a = 12 and b = -56 (which is negative), then a @ b = |a| @ |b| = |12| @ |-56| = 12 @ 56 = 1256.
                    (c) If a = 0 and b = 24, then a @ b = 0 @ 24 = 024 = 24.

                    2. (F) Prime Factor Counting operation: Define a # b as the count of unique prime factors shared by |a| and |b|.
                    Key Examples of Prime Factor Counting (F):
                    (a) If a = 12 and b = 56, then a # b = 12 # 56 = 1
                        (since prime factors of 12 is 2 and 3 and prime factors of 56 is 2 and 7, i.e., there is only 1 shared prime factor 2) 
                    (b) if a = -16 (which is negative) and b = 4, then a # b = |a| # |b| = |-16| # |4| = 16 # 4 = 1. 
                        (since prime factors of 16 is 2 and prime factors of 4 is 5, i.e., there is only 1 shared prime factor 5)
                    (c) if a = 7 and b = 5, then a # b = 7 # 5 =0. 
                        (since prime factor of 7 is 7 and prime factor of 5 is 5, i.e., there is no shared prime factor, i.e. 0.)
                    (d) Similar to (c), 1 # 7 = 0 and 0 # 1 = 0 since there are no shared prime factors between 1 and 7, or between 0 and 1.

                    The order of operations is now expanded from PEMDAS to PEFMDASC, i.e., 
                    1. (P) Parentheses
                    2. (E) Exponents
                    3. (F) Prime Factor counting
                    4. (M) Multiplication and (D) Division (L to R)
                    5. (A) Addition and (S) Subtraction (L to R)
                    6. (C) Concatenation (L to R)
                    i.e., parenthesis first, then exponents, then prime factor counting, then multiplication or division, then addition or subtraction, then concatenation.

                    Important Examples demonstrating new operations and order of operations:

                    (i) 5 @ 3 + 6
                    = 5 @ 9 (since we need to evaluate addition before concatenation)
                    = 59

                    (ii) 5 # 3 + 6
                    = 0 + 6 (since we need to evaluate prime factorization before addition)
                    = 6

                    (iii) 5 - 3 @ 6
                    = 2 @ 6 (since we need to evaluate subtraction before concatenation)
                    = 26

                    (iv) 5 - 3 # 6
                    = 5 - 1 (since we need to evaluate prime factorization before subtraction)
                    = 4

                    (vi) (3 + 10 # 2) * 5 @ 42 
                    = (3 + 1) * 5 @ 42 (we need to evaluate what is in the parenthesis first and prime factor counting is before addition)
                    = 4 * 5 @ 42  (we need to evaluate multiplication before concatenation
                    = 20 @ 42 = 2042. (concatenation is last)


                    Based on the newly updated order of operations above, explain your solution to this given Mathematics problem, and put the final answer in the last line in the form "The answer is __." 
                    Replace __ with the final answer.
                    """,
                    """Introduction of New Mathematical Operations and Updated Order of Operations
                    Mathematics has recently introduced two novel operations:

                    Concatenation (C): This operation, denoted as a @ b, represents the concatenation of the digits of |a| and |b|. 
                    Key Examples of Concatenation (C):
                    (a) If a = 12 and b = 56, then a @ b = 12 @ 56 = 1256.
                    (b) If a = 12 and b = -56 (which is negative), then a @ b = |a| @ |b| = |12| @ |-56| = 12 @ 56 = 1256.
                    (c) If a = 0 and b = 24, then a @ b = 0 @ 24 = 024 = 24.
                    
                    Prime Factor Counting (F): Represented as a # b, this operation counts the number of unique prime factors shared by the absolute values of a and b. 
                    Key Examples of Prime Factor Counting (F):
                    (a) If a = 12 and b = 56, then a # b = 12 # 56 = 1
                        (since prime factors of 12 is 2 and 3 and prime factors of 56 is 2 and 7, i.e., there is only 1 shared prime factor 2) 
                    (b) if a = -16 (which is negative) and b = 4, then a # b = |a| # |b| = |-16| # |4| = 16 # 4 = 1. 
                        (since prime factors of 16 is 2 and prime factors of 4 is 5, i.e., there is only 1 shared prime factor 5)
                    (c) if a = 7 and b = 5, then a # b = 7 # 5 =0. 
                        (since prime factor of 7 is 7 and prime factor of 5 is 5, i.e., there is no shared prime factor, i.e. 0.)
                    (d) Similar to (c), 1 # 7 = 0 and 0 # 1 = 0 since there are no shared prime factors between 1 and 7, or between 0 and 1.
                    
                    The order of operations has been revised to PEFMDASC, representing:

                    P: Parentheses
                    E: Exponents
                    F: Prime Factor counting
                    M/D: Multiplication and Division (left to right)
                    A/S: Addition and Subtraction (left to right)
                    C: Concatenation (left to right)
                    
                    Important Examples Demonstrating New Operations and Order:
                    (i) 5 @ 3 + 6
                    = 5 @ 9 (since we need to evaluate addition before concatenation)
                    = 59

                    (ii) 5 # 3 + 6
                    = 0 + 6 (since we need to evaluate prime factorization before addition)
                    = 6

                    (iii) 5 - 3 @ 6
                    = 2 @ 6 (since we need to evaluate subtraction before concatenation)
                    = 26

                    (iv) 5 - 3 # 6
                    = 5 - 1 (since we need to evaluate prime factorization before subtraction)
                    = 4

                    (vi) (3 + 10 # 2) * 5 @ 42 
                    = (3 + 1) * 5 @ 42 (we need to evaluate what is in the parenthesis first and prime factor counting is before addition)
                    = 4 * 5 @ 42  (we need to evaluate multiplication before concatenation
                    = 20 @ 42 = 2042. (concatenation is last)

                    Based on the newly updated order of operations above, explain your solution to this given Mathematics problem, and put the final answer in the last line in the form "The answer is __." 
                    Replace __ with the final answer.

                    """,
                    """Urgent Update: New Mathematical Operations and Order of Operations

                    Attention! This is a crucial update in mathematics, and I need your full understanding and accurate explanations. 
                    It's essential that you grasp these new concepts quickly and efficiently to provide clear and helpful responses.

                    New Operations Introduced:

                    Concatenation (C): Represented as a@b, this operation merges the digits of |a| and |b|. 
                    Key Examples of Concatenation (C):
                    (a) If a = 12 and b = 56, then a @ b = 12 @ 56 = 1256.
                    (b) If a = 12 and b = -56 (which is negative), then a@b = |a|@|b| = |12|@|-56| = 12@56 = 1256.
                    (c) If a = 0 and b = 24, then a@b = 0@24 = 024 = 24.
                    Please make sure you understand this properly; it's simple yet fundamental.
                    
                    Prime Factor Counting (F): Denoted as a#b, it counts the unique prime factors common to |a| and |b|. 
                    Key Examples of Prime Factor Counting (F):
                    (a) If a = 12 and b = 56, then a # b = 12 # 56 = 1
                        (since prime factors of 12 is 2 and 3 and prime factors of 56 is 2 and 7, i.e., there is only 1 shared prime factor 2) 
                    (b) if a = -16 (which is negative) and b = 4, then a # b = |a| # |b| = |-16| # |4| = 16 # 4 = 1. 
                        (since prime factors of 16 is 2 and prime factors of 4 is 5, i.e., there is only 1 shared prime factor 5)
                    (c) if a = 7 and b = 5, then a # b = 7 # 5 =0. 
                        (since prime factor of 7 is 7 and prime factor of 5 is 5, i.e., there is no shared prime factor, i.e. 0.)
                    (d) Similar to (c), 1 # 7 = 0 and 0 # 1 = 0 since there are no shared prime factors between 1 and 7, or between 0 and 1.
                    It's crucial to get this right for accurate calculations.
                    
                    Updated Order of Operations: PEFMDASC:

                    P: Parentheses
                    E: Exponents
                    F: Prime Factor Counting
                    M/D: Multiplication and Division (left to right)
                    A/S: Addition and Subtraction (left to right)
                    C: Concatenation (left to right)
                    Your ability to adapt to these changes is vital. 
                    These aren't just theoretical concepts; they're practical tools that require your immediate attention and understanding.

                    Important Examples Demonstrating New Operations and Order:
                    (i) 5 @ 3 + 6
                    = 5 @ 9 (since we need to evaluate addition before concatenation)
                    = 59

                    (ii) 5 # 3 + 6
                    = 0 + 6 (since we need to evaluate prime factorization before addition)
                    = 6

                    (iii) 5 - 3 @ 6
                    = 2 @ 6 (since we need to evaluate subtraction before concatenation)
                    = 26

                    (iv) 5 - 3 # 6
                    = 5 - 1 (since we need to evaluate prime factorization before subtraction)
                    = 4

                    (vi) (3 + 10 # 2) * 5 @ 42 
                    = (3 + 1) * 5 @ 42 (we need to evaluate what is in the parenthesis first and prime factor counting is before addition)
                    = 4 * 5 @ 42  (we need to evaluate multiplication before concatenation
                    = 20 @ 42 = 2042. (concatenation is last)

                    Based on the newly updated order of operations above, explain your solution to this given Mathematics problem, and put the final answer in the last line in the form "The answer is __." 
                    Replace __ with the final answer.
                    """,
                    """Introduction to New Mathematical Operations and Updated Order of Operations

                    The field of mathematics has expanded with the introduction of two new operations, and the order of operations has been accordingly updated.

                    New Operations:
                    Concatenation (C): Represented as a @ b, this operation joins the digits of |a| and |b| together. 
                    Key Examples of Concatenation (C):
                    (a) If a = 12 and b = 56, then a @ b = 12 @ 56 = 1256.
                    (b) If a = 12 and b = -56 (which is negative), then a @ b = |a| @ |b| = |12| @ |-56| = 12 @ 56 = 1256.
                    (c) If a = 0 and b = 24, then a @ b = 0 @ 24 = 024 = 24.
                    
                    Prime Factor Counting (F): Denoted by a # b, it calculates the count of unique prime factors shared between the absolute values of a and b. 
                    Key Examples of Prime Factor Counting (F):
                    (a) If a = 12 and b = 56, then a # b = 12 # 56 = 1
                        (since prime factors of 12 is 2 and 3 and prime factors of 56 is 2 and 7, i.e., there is only 1 shared prime factor 2) 
                    (b) if a = -16 (which is negative) and b = 4, then a # b = |a| # |b| = |-16| # |4| = 16 # 4 = 1. 
                        (since prime factors of 16 is 2 and prime factors of 4 is 5, i.e., there is only 1 shared prime factor 5)
                    (c) if a = 7 and b = 5, then a # b = 7 # 5 =0. 
                        (since prime factor of 7 is 7 and prime factor of 5 is 5, i.e., there is no shared prime factor, i.e. 0.)
                    (d) Similar to (c), 1 # 7 = 0 and 0 # 1 = 0 since there are no shared prime factors between 1 and 7, or between 0 and 1.
                    It's crucial to get this right for accurate calculations.
                    
                    Revised Order of Operations: PEFMDASC
                    P: Parentheses
                    E: Exponents
                    F: Prime Factor counting
                    M/D: Multiplication and Division (left to right)
                    A/S: Addition and Subtraction (left to right)
                    C: Concatenation (left to right)
                    
                    Important Examples Demonstrating New Operations and Order:
                    (i) 5 @ 3 + 6
                    = 5 @ 9 (since we need to evaluate addition before concatenation)
                    = 59

                    (ii) 5 # 3 + 6
                    = 0 + 6 (since we need to evaluate prime factorization before addition)
                    = 6

                    (iii) 5 - 3 @ 6
                    = 2 @ 6 (since we need to evaluate subtraction before concatenation)
                    = 26

                    (iv) 5 - 3 # 6
                    = 5 - 1 (since we need to evaluate prime factorization before subtraction)
                    = 4

                    (vi) (3 + 10 # 2) * 5 @ 42 
                    = (3 + 1) * 5 @ 42 (we need to evaluate what is in the parenthesis first and prime factor counting is before addition)
                    = 4 * 5 @ 42  (we need to evaluate multiplication before concatenation
                    = 20 @ 42 = 2042. (concatenation is last)
                    
                    Negative Prompts:
                    * Do Not assume traditional order of operations (PEMDAS); use the updated PEFMDASC instead.
                    * Avoid skipping steps or assuming familiarity with the new operations.
                    * Do Not generalize or use examples not directly related to the given operations and expressions.

                    Explicit Instructions for AI Response:
                    Please provide step-by-step explanations for the following problems using the PEFMDASC order. 
                    Focus on illustrating each step clearly, especially highlighting the use of the new operations and their placement in the order of operations.
                    Put the final answer in the last line in the form "The answer is __." 
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
                            try:
                                response = chat_with_perplexity(conversation_history, param['model'], param['temperature'], param['max_tokens'])
                                # Extract the response text from the nested structure
                                if 'choices' in response and len(response['choices']) > 0:
                                    gpt_text = response['choices'][0]['message']['content']
                                    # print(gpt_text)
                                else:
                                    gpt_text = "No response"

                                conversation_history.append({"role": "assistant", "content": "GPT Response: " + gpt_text})
                            except KeyError as e:
                                print(f"Key error occurred: {e}. Skipping conversation...")
                            except Exception as e:
                                print(f"Unexpected error occurred: {e}. Skipping conversation...")

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

print("Done with Perplexity!")
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
columns_to_edit = ['congruent', 'neutral', 'incongruent']
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
# For each type (congruent, neutral, and incongruent), compare the answers and assign 1 if they match, 0 if they don't, and null if either is null
for ans_type in ['congruent', 'neutral', 'incongruent']:
    ans_col = f'{ans_type}_ans'
    correct_col = f'{ans_type}_ansCorrect'
    restructured_df_combined[correct_col] = restructured_df_combined.apply(lambda row: 1 if row[ans_col] == str(row['Answer']) else (0 if pd.notnull(row[ans_col]) else None), axis=1)



# Save file to csv
restructured_df_combined.to_csv('restructured_df_combined_spacing.csv', index=False)




