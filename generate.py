from model import Transformer
from tokenizer import Tokenizer
from typing import Optional, Tuple, List
import time
from pathlib import Path
import json
from logging import getLogger

import torch
import torch.nn.functional as F

logger = getLogger("__name__")

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        device: str = "mps",
        float_type = torch.FloatTensor) -> "Llama":
        
        assert model_parallel_size is None, "This version doesn't support model parallel"
            
        torch.manual_seed(42)

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"No checkpoints in {ckpt_dir}"
        assert len(checkpoints) == 1, f"More than one checkpoint found in {ckpt_dir}, this version of llama only supports one checkpoint"
        
        ckpt_path = checkpoints[0] # we only get the first... cause there should only be one.
        checkpoint = torch.load(ckpt_path, map_location="cpu")  # we can probably load to gpu here, because there is only one...

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        params["max_seq_len"] = max_seq_len
        params["max_batch_size"] = max_batch_size
        params["ffn_dim_multiplier"] = None


        model_args = params # because it makes things easier

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args["vocab_size"] = tokenizer.n_words

        logger.info(f"model_args: {model_args}")
        torch.set_default_tensor_type(float_type)
        model = Transformer(**model_args)

        logger.info(f"state_dict_map: {list(checkpoint.keys())}")

        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        logger.info(f"unexpected_keys: {unexpected}")
        logger.info(f"missing_keys: {missing}")
        model.to(device=device)
        print(f"loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer, device)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, device: Optional[str] = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int, 
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False
        ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
            bsz = len(prompt_tokens)
            assert bsz <= self.model.max_batch_size, f"batch size too large: ({bsz},{self.model.max_batch_size})"
            
            min_prompt_len = min(len(t) for t in prompt_tokens)
            max_prompt_len = max(len(t) for t in prompt_tokens)
            assert max_prompt_len <= self.model.max_seq_len
            # figure out what the longest sequence we are expecting is.
            total_len = min(self.model.max_seq_len, max_gen_len + max_prompt_len)


            pad_id = self.tokenizer.pad_id
            # we create a tensor filled with the pad_id for our prompt/output
            tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
            for k, t in enumerate(prompt_tokens):
                # and pack it with the batch of prompts
                tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
            if logprobs:
                # if we are looking for the logprobs, we create an output tensor filled with zeros for them
                token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

            
            prev_pos = 0
            eos_reached = torch.tensor([False] * bsz, device=self.device)
            input_text_mask = tokens != pad_id

            for cur_pos in range(min_prompt_len, total_len):
                logger.info(f"tokens: {tokens.shape}, {tokens}")
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
                logger.info(f"logits: {logits.shape}, {logits[:, -1 ,:300]}")
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    logger.info(f"probs: {probs.shape}, {probs}[-1,:300]")
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)
                logger.info(f"next_token: {next_token.tolist()}")
                logger.info(f"next_token: {self.tokenizer.decode(next_token.tolist())}")
                
                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)

                tokens[:, cur_pos] = next_token
                if logprobs:
                    token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                        input = logits.transpose(1,2),
                        target = tokens[:, prev_pos + 1 : cur_pos + 1],
                        reduction = "none",
                        ignore_index = pad_id)
                    logger.info(f"logprobs: {token_logprobs.shape}, {token_logprobs}")
                eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
                prev_pos = cur_pos
                if all(eos_reached):
                    break

            if logprobs:
                token_logprobs = token_logprobs.tolist()

            out_tokens, out_logprobs = [], []

            for i, toks in enumerate(tokens.tolist()):
                start = 0 if echo else len(prompt_tokens[i])
                toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
                probs = None
                if logprobs:
                    probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
                # cut to eos if any
                if self.tokenizer.eos_id in toks:
                    eos_idx = toks.index(self.tokenizer.eos_id)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                out_tokens.append(toks)
                out_logprobs.append(probs)
            return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
                        

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    logger.info(f"top 10 indexes: {probs_idx[:,:10]}")
    logger.info(f"top 10 temperature probs: {probs_sort[:,:10]}")
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
