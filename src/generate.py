from llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()
vocab = model.get_path_to_vocab_file()
with open(vocab) as f:
    content = json.load(f)
id_to_token = {k:v for k,v in content.items()}
# print(vocab)
# print(list(content.items())[:10])

with open("data/input/functions_definition.json") as f:
    data = json.load(f)

# prompt = f"""Available function:{json.dumps(data)}
# question:1+1?
# Answers with a JSON object containing "name" and "parameters":
# """
with open("data/input/function_calling_tests.json") as f:
    prompte= json.load(f)
prompt = f"""Available function:{json.dumps(data)}
question = {prompte}
Answers with a JSON object containing "name" and "parameters":
"""

print(prompt)
id_tensor = model.encode(prompt)
# print("tensor: ",id_tensor)
id_list = id_tensor[0].tolist()
# print("liste: ", id_list)
tokken_text = []
print("wait...")
for i in range(100):
    logits = model.get_logits_from_input_ids(id_list)
    next_id = max(range(len(logits)), key=lambda i: logits[i])
    tokken_text.append(next_id)
    id_list.append(next_id)
    
print("generate")
print(model.decode(tokken_text))
