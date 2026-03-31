from transformers import GPT2Tokenizer, GPT2LMHeadModel

from beamspy import BeamSpy

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "Mary had a"
input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

# In this example we only have one prompt, so batch length will be 1. In general, you will have to set this up depending on your batch size!
# For "production" code you should avoid print_beams=True as it will slow generation down a lot when texts get long. We print beams here only for illustration/debug purposes.
peekAtBeams = BeamSpy(1, tokenizer, print_recipes=True, print_beams=True)

# STRONG recommendation: whenever you process logits you should renormalize them after, or else beam search (which assumes the logits describe a valid probability distribution) may become confused.
model.generate(input_ids, max_length = 50, num_beams = 4, renormalize_logits = True, logits_processor=[peekAtBeams])

