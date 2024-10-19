import torch
from model import decoder

import tiktoken

model_path = 'model_4000.pt'
device ='mps' if torch.backends.mps.is_available() else 'cpu'

enc = tiktoken.encoding_for_model("gpt-2")
vocab_size = enc.n_vocab

def encode(l):
    return enc.encode(l)


model = decoder(vocab_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model.to(device)

text = "Hi, could you translate it for me: Hello World!" # input what your want
context = torch.tensor(encode(text), dtype=torch.long, device=device).unsqueeze(0)


print(f"prompt:", text)
print('------------------------------------------------------')
print(f"generation:")
model.generate(context, max_new_tokens=500)
