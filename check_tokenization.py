
from transformers import AutoTokenizer

model_name = 'flaubert/flaubert_base_cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

premise = "Le chat est sur le tapis."
hypothesis = "Il y a un animal sur le tapis."

inputs = tokenizer(premise, hypothesis, return_token_type_ids=True)
decoded = tokenizer.decode(inputs['input_ids'])

print(f"Input IDs: {inputs['input_ids']}")
print(f"Token Type IDs: {inputs.get('token_type_ids', 'Not returned')}")
print(f"Decoded: {decoded}")

# Expected for FlauBERT: <s> premise </s> hypothesis </s> (or similar separator usage)
# FlauBERT usually uses: <s> S1 </s> </s> S2 </s>
