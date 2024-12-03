from openai import OpenAI

client = OpenAI(
    api_key=""
    ) 

returned = client.files.create(
                file=open("fine_tuning_dataset.jsonl", "rb"),
                purpose="fine-tune"
            )

print(returned)