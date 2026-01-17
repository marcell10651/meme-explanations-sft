from openai import OpenAI

client = OpenAI(
    base_url="",
    api_key=""
)

chat_completion = client.chat.completions.create(
    model="zai-org/GLM-4.5-Air-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is the purpose of life"}
    ],
    stream=False
)

print(chat_completion.choices[0].message.content)
