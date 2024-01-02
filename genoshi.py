import pandas as pd
import numpy as np
from openpyxl import load_workbook
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pymongo import MongoClient
import json

def excel_to_df(file):
    
    df = pd.read_excel(file)
    return df

def df_to_excel(df, file):
   
    df.to_excel(file, index=False)

def fetch_transcriptions(mongodb_uri):
    
    client = MongoClient(mongodb_uri)
    db = client.get_database()
    collection = db.papers
    cursor = collection.find({}, {"_id": 0, "transcription": 1})
    transcriptions = [item["transcription"] for item in cursor]
    return transcriptions

def ask_model(model, tokenizer, question, context):
    
    inputs = tokenizer.encode_plus(question, context, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = np.argmax(outputs.start_logits)
    answer_end = np.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

def fill_data(df, transcriptions):
   
    for i, row in df.iterrows():
        for j, value in row.iteritems():
            if pd.isna(value):
                row_header = df.columns[j]
                context = " ".join([str(df.loc[i][col]) for col in df.columns if not pd.isna(df.loc[i][col])])
                question = f"What is the {row_header}?"
                answer = ask_model(model, tokenizer, question, context)
                df.at[i, row_header] = answer
    return df


Input_Excel_Sheet = "https://docs.google.com/spreadsheets/d/1jOS2dunMCAoFTpZOyYedyG1mCMMGqcr9HB1tEh8usKI/edit#gid=0"
excel_file = "Input_Excel_Sheet"
df = excel_to_df(excel_file)

mongodb_uri = "mongodb+srv://intern:JeUDstYbGTSczN4r@interntest.i7decv0.mongodb.net/"
transcriptions = fetch_transcriptions(mongodb_uri)

model_name = "distilbert-base-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = fill_data(df, transcriptions)


output_excel_file = "Genoshi Intern Test - Output Excel Sheet"
df_to_excel(df, output_excel_file)