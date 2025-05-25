from __future__ import annotations

import math
import argparse
from collections import Counter
from datetime import datetime
from tqdm import tqdm

import torch
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True

from transformers import GPT2LMHeadModel

from model import DiT, CategoricalFlowMatching, SmallConfig, MediumConfig, LargeConfig


def get_curr_str_timestamp() -> str:
    return '_'.join(str(datetime.now()).split())


def _compute_sample_entropy(sample: list) -> float:
    histogram = Counter(sample)
    total = sum(histogram.values())
    entropy = 0
    for count in histogram.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def compute_entropy(samples: torch.LongTensor['B T']) -> torch.FloatTensor:
    entropies = [_compute_sample_entropy(sample.tolist()) for sample in samples]
    entropy = sum(entropies) / len(entropies)
    return torch.tensor(entropy, device=samples.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt_path', required=True, type=str, help='model checkpoint path')
    parser.add_argument('--config', required=False, default='small', type=str, help='model config name: small, medium or large')
    parser.add_argument('--length', required=False, default=1024, type=int, help='sequence length')
    parser.add_argument('--timesteps', required=False, default=1024, type=int, help='number of solver timesteps to use for generation')
    parser.add_argument('--topk', required=False, default=None, type=int, help='k for top-k tokens sampling')
    parser.add_argument('--temperature', required=False, default=1.0, type=float, help='sampling temperature')
    parser.add_argument('--bsz', '-b', required=False, default=4, type=int, help='batch size for evalution')
    parser.add_argument('--device', required=False, default='cuda', type=str, help='device to run compute on')
    parser.add_argument('--torch_compile', action='store_true', default=False, help='compile model for faster inference')
    parser.add_argument('--num_samples', required=False, default=1024, type=int, help='number of samples for eval')
    parser.add_argument('--eval_entropy', action='store_true', default=False, help='compute generative entropy of the model')
    parser.add_argument('--eval_perplexity', action='store_true', default=False, help='evaluate generative perplexity or not')
    parser.add_argument('--eval_hellaswag', action='store_true', default=False, help='evaluate performance on `hellaswag` dataset')
    parser.add_argument('--eval_lambada', action='store_true', default=False, help='evaluate performance on `lambada` dataset')
    parser.add_argument('--seed', required=False, default=1234, type=int, help='random seed')
    args = parser.parse_args()
    
    checkpt_path = args.checkpt_path
    config_name = args.config
    length = args.length
    timesteps = args.timesteps
    topk = args.topk
    temperature = args.temperature
    bsz = args.bsz
    device = args.device
    torch_compile = args.torch_compile
    num_samples = args.num_samples
    eval_entropy = args.eval_entropy
    eval_perplexity = args.eval_perplexity
    eval_hellaswag = args.eval_hellaswag
    eval_lambada = args.eval_lambada
    seed = args.seed
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    str_timestamp = get_curr_str_timestamp()
    
    def print_(s: str) -> None:
        print('>>>', s)
    
    # initialize model
    config = {'small': SmallConfig, 'medium': MediumConfig, 'large': LargeConfig}[config_name]()
    dit = DiT(config.dim, config.n_heads, config.dim_mult, config.n_layers, config.vocab_size)
    model = CategoricalFlowMatching(dit, vocab_size=config.vocab_size, eos_idx=config.eos_idx)
    model = model.eval().to(device)
    print_('model initialized')
    
    # load checkpoint
    checkpt = torch.load(checkpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict({k.replace('._orig_mod', ''): v for k, v in checkpt['model'].items()})
    print_('checkpoint loaded')
    
    # compiling model if required
    if torch_compile:
        model.compile(mode='default', fullgraph=True, dynamic=True)
        print_('model compiled')
    
    fstream = open(f'eval_{str_timestamp}.txt', 'w')
    fstream.writelines([
        'model: llada\n',
        f'checkpoint path: {checkpt_path}\n',
        f'training iteration: {checkpt['iteration']}\n'
    ])
    
    if eval_entropy or eval_perplexity:
        # we need to generate `num_samples` samples with our trained model
        n_batches = num_samples // bsz
        samples = []
        with torch.amp.autocast(device, dtype=torch.bfloat16), torch.no_grad():
            for _ in tqdm(range(n_batches), desc='generating samples for perplexity evaluation'):
                x1 = model.sample(
                    bsz, T=length,
                    timesteps=timesteps,
                    topk=topk,
                    temperature=temperature,
                    verbose=False,
                    parse_outputs=False
                )[0]  # [B, T]
                samples.append(x1)
            samples = torch.cat(samples, 0)  # [num_samples, T]
            torch.save(samples, f'eval_samples_{str_timestamp}.pt')
    
    if eval_entropy:
        entropy = compute_entropy(samples)
        print_(f'entropy = {entropy}')
        fstream.writelines([
            '--------------------------\n',
            'ENTROPY EVAL:\n'
            f'num samples: {num_samples}\n',
            f'entropy: {entropy.item()}\n'
        ])
    
    if eval_perplexity:
        # initializing gpt2-large as eval model
        # @TODO: use Llama-3
        eval_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device).eval()
        print_('gpt2-large initialized')
        
        # computing generative perplexity with eval_model
        ppls = []
        with torch.no_grad():
            for i in tqdm(range(n_batches), desc='estimating perplexity'):
                x = samples[i*bsz:(i+1)*bsz]  # [B, T]
                out = eval_model(input_ids=x, labels=x)  # [B, T, vocab_size]
                ppls.append(out.loss.exp())
        perplexity = torch.stack(ppls).mean()
        
        print_(f'generative perplexity (via `gpt2-large`) = {perplexity}')
        fstream.writelines([
            '--------------------------\n',
            'GENERATIVE PERPLEXITY EVAL:\n'
            f'num samples: {num_samples}\n',
            'eval_model: gpt2-large\n',
            f'perplexity: {perplexity.item()}\n'
        ])
    
    fstream.close()
