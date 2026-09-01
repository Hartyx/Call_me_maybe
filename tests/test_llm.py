import json
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
vocab_path = model.get_path_to_vocab_file()

with open(vocab_path, "r") as f:
    vocab = json.load(f)

print(f"Nombre total de tokens : {len(vocab)}")
print("Type de structure :", type(vocab))

items = list(vocab.items())
for item in items:
    print(item)
with open("vocab_dump.txt", "w", encoding="utf-8") as out:
    json.dump(vocab, out, ensure_ascii=False, indent=2)
print("Vocabulaire complet sauvegardé dans vocab_dump.json")