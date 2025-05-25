import os
import numpy as np
import argparse
from collections import OrderedDict
from time import perf_counter

import torch
from torch.utils.tensorboard import SummaryWriter
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True
torch._inductor.config.coordinate_descent_tuning = True
from accelerate import Accelerator

from data import StreamingDataset, PretrainBatchCollator, SFTBatchCollator
from model import DiT, CategoricalFlowMatching, SmallConfig, MediumConfig, LargeConfig


def _get_attr(obj, names: list[str]):
    if len(names) == 1:
        return getattr(obj, names[0])
    return _get_attr(getattr(obj, names[0]), names[1:])


def load_checkpt(
    model: torch.nn.Module,
    state_dict: OrderedDict,
    strict: bool = True,
    allow_size_mismatches: bool = False
) -> list[tuple[str, str, torch.Size, torch.Size]]:
    if strict:
        return model.load_state_dict(state_dict, strict=True)
    invalid_keys = []
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        try:
            value_curr = _get_attr(model, key.split('.'))
            if value_curr.shape != value.shape:
                if allow_size_mismatches:
                    value_to_set = value_curr
                    invalid_keys.append(('mismatch', key, value_curr.shape, value.shape))
                else:
                    msg = (
                        f'Bad boy! For weight `{key}` checkpoint contains the different size! '
                        f'Got `{value.shape}`, but needed `{value_curr.shape}`. '
                        'Maybe you wanted to set `allow_size_mismatches=True`?'
                    )
                    raise RuntimeError(msg)
            else:
                value_to_set = value
            new_state_dict[key] = value_to_set
        except AttributeError:
            invalid_keys.append(('not-exist', key))
    model.load_state_dict(new_state_dict, strict=False)
    return invalid_keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=False, default='./logs', type=str, help='base log directory path')
    parser.add_argument('--name', required=True, type=str, help='experiment name')
    parser.add_argument('--config', required=False, default='small', type=str, help='model config name: small, medium or large')
    parser.add_argument('--mode', '-m', required=True, type=str, help='one of the training modes: [pretrain, sft]')
    parser.add_argument('--bsz', '-b', required=False, default=32, type=int, help='batch size for training')
    parser.add_argument('--lr', required=False, default=1e-4, type=float, help='learning rate')
    parser.add_argument('--n_epochs', required=False, default=10, type=int, help='number of epochs')
    parser.add_argument('--torch_compile', action='store_true', default=False, help='compile model for faster training')
    parser.add_argument('--dtype', required=False, default='bf16', help='data type of tensors for mixed-precision')
    parser.add_argument('--keep_latest_every', required=False, default=1000, type=int, help='number of iterations to update the latest checkpt')
    parser.add_argument('--snapshot_every', required=False, default=25_000, type=int, help='number of iterations to make a snapshot')
    parser.add_argument('--continue_training', action='store_true', default=False, help='whether to continue an experiement')
    parser.add_argument('--reset_opt', action='store_true', default=False, help='whether to reset optimizer if continuing experiment')
    args = parser.parse_args()
    
    logdir = args.logdir
    name = args.name
    config_name = args.config
    log_path = f'{logdir}/{name}'
    mode = args.mode
    bsz = args.bsz
    lr = args.lr
    n_epochs = args.n_epochs
    torch_compile = args.torch_compile
    dtype = args.dtype
    keep_latest_every = args.keep_latest_every
    snapshot_every = args.snapshot_every
    continue_training = args.continue_training
    reset_opt = args.reset_opt
    
    config = {'small': SmallConfig, 'medium': MediumConfig, 'large': LargeConfig}[config_name]()
    
    accelerator = Accelerator(mixed_precision=dtype)
    device = accelerator.device
    grank = accelerator.process_index
    is_rank0 = grank == 0
    world_size = accelerator.num_processes
    
    # fix all seed and set torch cudnn arguments
    np.random.seed(grank + 1234)
    torch.manual_seed(grank + 1234)
    torch.cuda.manual_seed(grank + 1234)
    torch.cuda.manual_seed_all(grank + 1234)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
    def print0(s: str) -> None:
        if is_rank0: print('>>>', s)
    
    print0(f'world size: {world_size}')
    print0(f'total bsz per gpu: {bsz}')
    print0(f'total bsz: {world_size * bsz}')
    
    if is_rank0:
        if continue_training:
            assert os.path.exists(log_path), f'kabzda, no experiment to continue from path `{log_path}`!'
        else:
            assert not os.path.exists(log_path), f'kabzda, experiment with this name ({name}) already exists!'
            os.makedirs(log_path)
            print0('log directory created')

    # initialize model
    dit = DiT(config.dim, config.n_heads, config.dim_mult, config.n_layers, config.vocab_size)
    model = CategoricalFlowMatching(dit, config.vocab_size, config.eos_idx)
    print0('model initialized')
    print0(f'num of parameters: {sum([p.numel() for p in model.parameters()])}')
    
    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print0('optimizer initialized')
    
    # load checkpoint if continuing training
    if continue_training:
        checkpt = torch.load(log_path + '/latest.pt', map_location='cpu', weights_only=True)
        state_dict = {k.replace('._orig_mod', ''): v for k, v in checkpt['model'].items()}
        invalid_keys = load_checkpt(model, state_dict, strict=False, allow_size_mismatches=True)
        if len(invalid_keys) > 0:
            print0(f'invalid keys in checkpoint: {invalid_keys}')
        if not reset_opt:
            optimizer.load_state_dict(checkpt['optimizer'])
        start_iteration, start_epoch = checkpt['iteration'], checkpt['epoch']
        print0('latest checkpoint loaded')
    else:
        start_iteration, start_epoch = 0, 0
    
    # compiling model if required
    if torch_compile:
        model.compile(mode='default', fullgraph=True, dynamic=True)
        print0('model compiled')
    
    # creating tensorboard logger
    logger = SummaryWriter(log_dir=log_path, purge_step=start_iteration)
    print0('logger initialized')
    
    # prepare data loading objects
    if mode == 'pretrain':
        dataset_init_kwargs = {
            'paths': ['HuggingFaceFW/fineweb'] * 3,
            'names': ['CC-MAIN-2024-51', 'CC-MAIN-2024-46', 'CC-MAIN-2024-42'],
            'splits': ['train'] * 3
        }
        collate_fn = PretrainBatchCollator(max_length=2048, pad_idx=config.pad_idx)
    elif mode == 'sft':
        dataset_init_kwargs = {'paths': ['TIGER-Lab/Fineweb-Instruct'], 'splits': ['train']}
        collate_fn = SFTBatchCollator(max_length=4096, pad_idx=config.pad_idx)
    else:
        raise KeyError(f'No mode `{mode}`.')
    dataset = StreamingDataset(grank, world_size, **dataset_init_kwargs)
    dataset.skip(start_iteration)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bsz, collate_fn=collate_fn, shuffle=False)
    print0('initialized data loader')
    
    torch.distributed.barrier(device_ids=[grank])
    
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    print0('prepared objects with accelerate')
    
    print0('running training loop:')
    iteration = start_iteration
    model.train()
    timings, t0 = [], perf_counter()
    for epoch in range(start_epoch, n_epochs):
        for tokens, _ in loader:
            tokens = tokens.to(device)
            optimizer.zero_grad()
            
            loss, accuracy, perplexity = model(tokens)
            accelerator.backward(loss)
            gn = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # we track average step time needed to make a single training step
            timing = (perf_counter() - t0) * 1000
            timings.append(timing)
            if len(timings) > 100:  # tracking only last 100 updates
                timings.pop(0)
            
            if is_rank0:
                if iteration % 50 == 0:
                    print((
                        f'epoch: {epoch}, '
                        f'iteration: {iteration}, '
                        f'loss: {round(loss.item(), 3)}, '
                        f'acc: {round(accuracy.item(), 3)}, '
                        f'gn: {round(gn.item(), 3)}, '
                        f'step: ~{round(np.mean(timings), 2)}ms'
                    ))
                logger.add_scalar('training/loss', loss, iteration)
                logger.add_scalar('training/accuracy', accuracy, iteration)
                logger.add_scalar('training/perplexity', perplexity, iteration)
                logger.add_scalar('training/grad_norm', gn, iteration)
                
                # keeps the latest checkpoint up-to-date
                if iteration % keep_latest_every == 0:
                    checkpt = {
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch
                    }
                    torch.save(checkpt, log_path + '/latest.pt')
                    print0(f'updated latest checkpoint at iteration={iteration}')
                
                # makes a separate snapshot of the model
                if iteration % snapshot_every == 0:
                    checkpt = {
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch
                    }
                    torch.save(checkpt, log_path + f'/checkpt-{iteration}.pt')
                    print0(f'saved snapshot at iteration={iteration}')
            
            t0 = perf_counter()
            iteration += 1
