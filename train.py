import os
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from data import StreamingDataset, PretrainBatchCollator, SFTBatchCollator
from model import DiT, CategoricalFlowMatching, LLaDAConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str, help='experiment name')
    parser.add_argument('--mode', '-m', required=True, type=str, help='one of the training modes: [pretrain, sft]')
    parser.add_argument('--bsz', '-b', required=False, default=32, type=int, help='batch size for training')
    parser.add_argument('--lr', required=False, default=1e-4, type=float, help='learning rate')
    parser.add_argument('--n_epochs', required=False, default=10, type=int, help='number of epochs')
    parser.add_argument('--torch_compile', action='store_true', default=False, help='compile model for faster training')
    args = parser.parse_args()
    
    name = args.name
    log_path = './logs/' + name
    mode = args.mode
    bsz = args.bsz
    lr = args.lr
    n_epochs = args.n_epochs
    
    config = LLaDAConfig()
    
    accelerator = Accelerator()
    device = accelerator.device
    grank = accelerator.process_index
    is_rank0 = grank == 0
    
    if is_rank0:
        assert not os.path.exists(log_path), f'kabzda, experiment with this name ({name}) already exists!'
        os.makedirs(log_path)
        logger = SummaryWriter(log_dir=log_path)

    # initialize model
    dit = DiT(config.dim, config.n_heads, config.dim_mult, config.n_layers, config.vocab_size)
    model = CategoricalFlowMatching(dit, config.vocab_size, config.eos_idx)
    
    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # prepare data loading objects
    if mode == 'pretrain':
        dataset_init_kwargs = {'path': 'HuggingFaceFW/fineweb', 'name': 'CC-MAIN-2024-10', 'split': 'train', 'streaming': True}
        collate_fn = PretrainBatchCollator(max_length=1024, pad_idx=config.pad_idx)
    elif mode == 'sft':
        dataset_init_kwargs = {'path': 'TIGER-Lab/Fineweb-Instruct', 'split': 'train', 'streaming': True}
        collate_fn = SFTBatchCollator(max_length=4096, pad_idx=config.pad_idx)
    else:
        raise KeyError(f'No mode `{mode}`.')
    dataset = StreamingDataset(grank, accelerator.num_processes, **dataset_init_kwargs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bsz, collate_fn=collate_fn, shuffle=False)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    iteration = 0
    model.train()
    for epoch in range(n_epochs):
        for tokens, _ in loader:
            tokens = tokens.to(device)
            optimizer.zero_grad()
            
            loss, accuracy = model(tokens)
            accelerator.backward(loss)
            gn = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if is_rank0:
                if iteration % 50 == 0:
                    print((
                        f'epoch: {epoch}, '
                        f'iteration: {iteration}, '
                        f'loss: {round(loss.item(), 3)}, '
                        f'acc: {round(accuracy.item(), 3)}, '
                        f'gn: {round(gn.item())}'
                    ))
                logger.add_scalar('training/loss', loss, iteration)
                logger.add_scalar('training/accuracy', accuracy, iteration)
            iteration += 1
