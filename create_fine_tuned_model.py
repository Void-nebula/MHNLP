from openai import OpenAI
client = OpenAI(
    api_key=""
    )

returned = client.fine_tuning.jobs.create(
    training_file="file-SgXz4GThAXhKRzPED9QwRq",
    model="gpt-4o-mini-2024-07-18"
)

print(returned)