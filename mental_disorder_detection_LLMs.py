import argparse
import pandas as pd
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from huggingface_hub import login
from sklearn.model_selection import train_test_split

client = OpenAI(
    api_key=""
    ) 

def pipeline_gpt(msg, client):
    response = client.chat.completions.create(
        # model="gpt-4o-mini",
        model="ft:gpt-4o-mini-2024-07-18:personal::AYioj1cS",
        messages=[
            {"role": "system", 
            "content": "You are a professional therapist. Your patient is telling you about their basic information and some answers to a mental health servey. Based on the information, please infer whether the patient has depression and/or anxiety disorders. 0: Neither, 1: Either or both. Based on this correspondence, you only need to answer the number."},

            {
                "role": "user",
                "content": msg
            }
        ],
        temperature=1.0,
        # top_k=1.0,
        top_p=1.0
    )

    return response.choices[0].message.content

def pipeline_causallm():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Configure the pipeline for Causal Language Modeling
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        trust_remote_code=True
    )

    # model_name = "/home/txie/Mental_Health/CS_584_Course_Project/MHNLP/fine_tuned"
    # tokenizer = AutoTokenizer.from_pretrained("/home/txie/Mental_Health/CS_584_Course_Project/MHNLP/fine_tuned")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     load_in_8bit=True,
    #     trust_remote_code=True
    # )
    
    pipeline = transformers.pipeline(
        task='text-generation', 
        model=model, 
        tokenizer=tokenizer, 
        device_map=device
    )

    return pipeline

def prep_prompt_classification_llama_desc(row):

    prompt = f"""

    <s> [INST] <<SYS>>
    You are a professional therapist. Your patient is telling you about their basic information and some answers to a mental health servey. Based on the information, please infer whether the patient has depression and/or anxiety disorders.
    
    Please answer with only one digit: 0, 1, where 0 means the patient does not have depression or anxiety disorders, and 1 means the patient has depression and/or anxiety disorders. For example, if the input indicates the patient don't have depression or anxiety disorder, your answer shuold be: 0. Otherwise, your answer should be: 1. You don't need to explain your answer.</s>
    <</SYS>>

    [/INST]

    Sure, let me know the responses that the patient said. I'll try to classify if they have depression and/or anxiety disorders. </s>

    </s>

    <s> [INST]

    The responses are as follows:

    {row["prompt"]}

    [/INST]

    I think the digit should be: </s>
    """

    return prompt

def clean_response(msg, client):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", 
            "content": ""},

            {
                "role": "user",
                "content": msg
            }
        ],
        temperature=1.0,
        # top_k=1.0,
        top_p=1.0
    )

    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()

    # Define the arguments
    parser.add_argument('--model', type=str, help='The model that will perform the task', required=True)
    parser.add_argument('--csv-path', type=str, help='the path of dataset', required=True)

    # Parse the arguments
    args = parser.parse_args()

    # GPT model
    if args.model == 'gpt':
        df = pd.read_csv(args.csv_path)
        results = []
        for _, row in df.iterrows():
            msg = row['prompt']
            response = pipeline_gpt(msg, client)
            results.append(int(response))
        # df['predict'] = results
        # df.to_csv('output.csv', index=False)

        y_pred = np.array(results)
        y_true = np.array(df['label'])

        accuracy = np.mean(y_pred == y_true)
        print(f"Accuracy: {accuracy}")
    
    # Llama model
    elif args.model == 'llama':
        pipeline_kwargs = {
            # 'max_length'=4096,
            # 'max_new_tokens' : 50,
            # 'min_length': 100,
            # 'early_stopping':True,
            'do_sample':True,
            # 'top_k':10,
            # 'top_p':1,
            'temperature':1,
            # 'num_return_sequences':1,
            # 'no_repeat_ngram_size': 2,
            'return_full_text':False,
            # 'eos_token_id':tokenizer.eos_token_id
            # 'max_length':2000,
        }
        pipeline = pipeline_causallm()
        df = pd.read_csv(args.csv_path)
        inputs = []
        results = []
        for _, row in df.iterrows():
            inputs.append(prep_prompt_classification_llama_desc(row))
        answers = pipeline(inputs, **pipeline_kwargs)
        for answer in answers:
            response = answer[0]['generated_text'][0]
            results.append(int(response))

        y_pred = np.array(results)
        y_true = np.array(df['label'])

        accuracy = np.mean(y_pred == y_true)
        print(f"Accuracy: {accuracy}")

    else:
        raise ValueError(f"Model {args.model} not found")

if __name__ == '__main__':
    main()