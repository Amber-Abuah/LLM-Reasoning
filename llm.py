import ollama

model_name = "llama3"

def ai_chat(prompt):
    return ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])["message"]["content"]

def ai_chat_chain(prompts):
    message_chain = []

    for i in range(len(prompts)):
        message_chain.append({"role": "user", "content": prompts[i]})
        message_chain.append(
            {"role": "assistant", "content": ollama.chat(model=model_name, messages=message_chain)["message"]["content"]})
        
    return message_chain[-1]["content"]
