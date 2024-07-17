# %%time
import openai
# import pdfplumber
import json
import time
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import sys
import subprocess
import csv
import logging 
import pandas as pd 
import os
import psycopg2
from datetime import datetime
import asyncio
from time import sleep
import re
from img2table.document import PDF
from img2table.ocr import TesseractOCR
from IPython.display import display_html
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openpyxl
import PyPDF2

start_time = time.time()
list_value_tables = []
list_keys_tables= []

# Set your OpenAI API key
start_time = time.time()
# start_time = time.time.now()
openai.api_key = '' #Enter your key here

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader (file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text ()
    return text

# messages=[
#             {"role":"system","content":"You are Key value maker assistant which generate key value pairs of the tables given in the text."},
#             {"role":"user","content": prompt}
#         ]
#     )



# Function to generate key-value pairs using OpenAI
def generate_key_value_pairs(prompt):
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
#         seed=0,
        messages=[
            {"role":"system","content":"You are Key value maker assistant which generate key value pairs of the tables given in the text."},
            {"role":"user","content": prompt}
        ]
    )
    return completion.choices[0].message.content,completion

def correct_multiline_string_format(incorrect_multiline_string):
    prompt = f"Correct the format of the following JSON string:\n\n{incorrect_multiline_string}\n\nAfter correction, provide the properly formatted JSON:"
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content,completion

print(time.time() - start_time)

# Replace 'your-pdf-file.pdf' with the path to your PDF file

# Generate key-value pairs using the combined prompt
def key_value_main_function(pdf_text):
    prompt = f"Given the following text from the PDF:\n\n{pdf_text}\n\nGenerate key-value pairs in dictionary format, and don't summarize and paraphrase it."
    result,completion_res = generate_key_value_pairs(prompt)

    # Use triple quotes to define the multiline string
    result_multiline = f'''{result}'''
    corrected_multiline,corrected_multiline_res=correct_multiline_string_format(result_multiline)
#     print(corrected_multiline)
#Convert the multiline string to a JSON object
#     json_object = json.loads(corrected_multiline)
    return corrected_multiline,corrected_multiline_res
# print(key_value_main_function(pdf_text))

pdf_dir_path = r'D:\Varishth_MNCFC\PDF\try_dir'
def extract_text_from_page(pdf_path, page_number):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader (file)
            if page_number < 1 or page_number >  len(reader.pages):
                raise ValueError("Invalid page number")
            page = reader.pages[page_number-1] # Page numbers are 0-indexed
            text = page.extract_text()
            return text
    except FileNotFoundError:
        print("Error: PDF file not found.")
    except Exception as e:
        print("Error: Unable to read PDF file.",e)
    except ValueError as e:
        print(f"Error: {e}")

# Example usage:
# pdf_path = r"D:\Varishth_MNCFC\PDF\Quote_Slips\Hollard\Transalloys_Placing Slip 1_20220701_signed TRS July 11 2022 (005).pdf"
def get_pdf_page_numbers(pdf_path):
    page_numbers = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_number in range(len(reader.pages)):
            page_numbers.append(page_number + 1)  # Pages are 0-indexed, so add 1
    return page_numbers

import re

def remove_elements_with_pattern(list_keys,list_value,pattern):
    filtered_list_keys = []
    filtered_list_values=[]
    for item in list_keys:
        if not re.match(pattern, item):
            filtered_list_keys.append(item)
            filtered_list_values.append(list_value[list_keys.index(item)])
            
    return filtered_list_keys,filtered_list_values

pattern = r"Table \d+<\|\|>.*"
import json

def store_keys(dictionary, parent_key=None):
    keys = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            subkeys = store_keys(value, key)
            keys.extend([f"{key}<||>{subkey}" for subkey in subkeys])
        else:
            keys.append(key)
    return keys

# Load JSON file
# with open(json_object, 'r') as f:
#     data = json.load(f)
# data = json_object

# Store all keys and their unique subkeys
# all_keys = store_keys(data)
# Print all keys and unique subkeys



def count_occurrences(lst):
    count = 0
    for item in lst:
        if isinstance(item, str):
            count += item.count("<||>")
    return count



def count_occurrences(main_string, sub_string):
    return main_string.count(sub_string)

def get_nested_value(data, keys):
    value = data
    for key in keys:
        value = value.get(key, None)
        if value is None:
            return None
    return value

def get_nested_value(data, keys):
    value = data
    for key in keys:
        if key in value:
            value = value[key]
        else:
            return None
    return value

# filtered_list_keys,filtered_list_values = remove_elements_with_pattern(list_keys,list_value,pattern)



def convert_dict_json(corrected_multiline):
    json_object = json.loads(corrected_multiline)
    r = json.dumps(json_object)
    loaded_r = json.loads(r)
    all_keys = store_keys(loaded_r)
    return all_keys,json_object
# page_numbers = get_pdf_page_numbers(pdf_path)
counter = 3211
qiid = 19
for i in os.listdir(pdf_dir_path):
    print(i)
    pdf_path = os.path.join(pdf_dir_path,i)
    
    page_numbers = get_pdf_page_numbers(pdf_path)
    for j in page_numbers:
        print(j)
        text_from_page = extract_text_from_page(pdf_path,int(j))
    #     key_value_main_function(text_from_page)
        corrected_multiline,corrected_multiline_res=key_value_main_function(text_from_page)
        corrected_multiline,corrected_multiline_res=correct_multiline_string_format(corrected_multiline)
        corrected_multiline,corrected_multiline_res=correct_multiline_string_format(corrected_multiline)
        corrected_multiline,corrected_multiline_res=correct_multiline_string_format(corrected_multiline)
    #     status = True
        #     start_time = time.time()
        while True:
            print("in while")
            try:
                print("in try------")
                all_keys,json_object = convert_dict_json(corrected_multiline)
                # list_keys_tables = [i for i in all_keys]
                for i in all_keys:
                    list_keys_tables.append(i)
                #     print(i)
                    value_extracted = ''
                    data_to_be_searched = i
                    substring = "<||>"
                    occurence=count_occurrences(i, substring)
                    splitted=''
                    if(occurence>0):
                        splitted=i.split("<||>")
                    if(len(splitted)>0):
                #         print(splitted)
                        value_extracted = get_nested_value(json_object, splitted)
                    else:
                        value_extracted=json_object[data_to_be_searched]
                    list_value_tables.append(value_extracted)
                print("----------")
                print('==INDEX==',page_numbers.index(j))
                break
            except Exception as e:
                print(e)
                corrected_multiline,corrected_multiline_res=correct_multiline_string_format(corrected_multiline)
    
    # Your database credentials
    hostserver = "localhost"
    database = "NewWorld_db_updated"
    username = "postgres"
    password_server = "pass@123"
    conn = psycopg2.connect(
            host=hostserver,
            database=database,
            user=username,
            password=password_server
        )
        

    
    cursor = conn.cursor()
    query_update_prediction = '''INSERT INTO pdf_output_5_sample(QID,Key,S_NO,Output) VALUES('%s','%s',%s,'%s')'''
    for i,j in zip(list_keys_tables,list_value_tables):
        y = query_update_prediction
        print(y)
        if (type(j)==str or type(j)==int or type(j)==bool or type(j)==None or type(j)==float or type(j)==dict):
            cursor.execute('''INSERT INTO pdf_output_5_sample(QID,Key,S_NO,Output) VALUES('{}','{}',{},'{}')'''.format(str(qiid),i.replace('\'','\'\''),counter,str(j).replace('\'','\'\'')))
    #     elif(type(j) == int):


        else:
            try:
                cursor.execute('''INSERT INTO pdf_output_5_sample(QID,Key,S_NO,Output) VALUES('{}','{}',{},'[{}]')'''.format(str(qiid),i.replace('\'','\'\''),counter,','.join(j).replace('\'','\'\'')))
            except:
                cursor.execute('''INSERT INTO pdf_output_5_sample(QID,Key,S_NO,Output) VALUES('{}','{}',{},'[{}]')'''.format(str(qiid),i.replace('\'','\'\''),counter,str(j).replace('\'','\'\'')))
        conn.commit()
        counter+=1
    cursor.close()
    qiid+=1

print(time.time() - start_time)