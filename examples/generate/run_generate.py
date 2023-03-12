import torch
from transformers import AutoTokenizer
from memformers.models.membart import MemBartForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
# load the large model
membart = MemBartForConditionalGeneration.from_pretrained("qywu/membart-large")


text1 = "Barack Obama served as the 44th President of the United States."
text2 = "<mask> served as the 44th President of the United States."

# construct the initial memory
memory_states = membart.construct_memory(batch_size=1)

# t = 0
input_ids1 = torch.LongTensor([tokenizer.encode(text1)])
# only run the encoder to get memory states
encoder_outputs = membart.model.encoder(input_ids=input_ids1, memory_states=memory_states, attention_mask=None)
memory_states = encoder_outputs.memory_states


# t = 1
input_ids2 = torch.LongTensor([tokenizer.encode(text2)])

encoder_outputs2 = membart.model.encoder(input_ids=input_ids2, memory_states=memory_states, attention_mask=None)

outputs = membart.generate(
    encoder_outputs=encoder_outputs2,
    decoder_start_token_id=tokenizer.bos_token_id,
    max_length=64,
    num_beams=1,
    do_sample=False,
    return_dict_in_generate=True,
)

print(tokenizer.decode(outputs.sequences[0]))
