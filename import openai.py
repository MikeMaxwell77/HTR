import openai

openai.api_key = "your-api-key-here"

def clean_text(messy_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that fixes messy, ungrammatical, or disorganized text."},
            {"role": "user", "content": f"Fix this text: {messy_text}"}
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# Example usage:
messy_input = "this is a very bad wrttin txt, that no make sense properly"
fixed_text = clean_text(messy_input)
print(fixed_text)
