from openai import OpenAI

client = OpenAI(
    api_key=""
)

# Retrieve the state of a fine-tune
returned = client.fine_tuning.jobs.retrieve("ftjob-RIErFxfJfAAXbBmJbmhEDB8x")

print(returned)