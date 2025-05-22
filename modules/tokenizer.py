
from transformers import GPT2TokenizerFast, DebertaV2Tokenizer, Qwen2TokenizerFast

def build_tokenizer_qwen(args):
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.tokenizer_name)
    # tokenizer.pad_token = 0
    return tokenizer

def build_tokenizer(args):
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_deberta_tokenizer(args):
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.tokenizer_name)
    return tokenizer
    