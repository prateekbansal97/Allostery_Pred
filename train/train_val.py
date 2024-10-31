import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from train.utils import kl_categorical, kl_categorical_uniform, nll_gaussian, edge_accuracy

var = 5e-5

def train(encoder, decoder, optimizer, scheduler, train_loader, valid_loader, rel_rec, rel_send, epoch, best_val_loss, ntimesteps, num_residues, log_prior, edge_types, sysname, device, log):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    edges_train = []
    probs_train = []

    encoder.train()
    decoder.train()

    for batch_idx, (data, relations) in enumerate(train_loader):

        data, relations = data.to(device), relations.to(device)

        optimizer.zero_grad()

        logits = encoder(data, rel_rec, rel_send)
        
        edges = F.gumbel_softmax(logits, tau=0.5, hard=True)
        prob = F.softmax(logits, -1)

        output = decoder(data, edges, rel_rec, rel_send, ntimesteps, burn_in=True, burn_in_steps=ntimesteps-1)
        
        target = data[:, 1:, :, :]
        target = target.transpose(2, 1)
        
        loss_nll = nll_gaussian(output, target, var)

        loss_kl = kl_categorical(prob, log_prior, num_residues)

        loss = loss_nll + loss_kl

        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)

        loss.backward()
        optimizer.step()

        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        _, edges_t = edges.max(-1)
        edges_train.append(edges_t.data.cpu().numpy())
        probs_train.append(prob.data.cpu().numpy())

    scheduler.step()
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    encoder.eval()
    decoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        data, relations = data.cuda(), relations.cuda()
        with torch.no_grad():

            logits = encoder(data, rel_rec, rel_send)
            edges = F.gumbel_softmax(logits, tau=0.5, hard=True)
            prob = F.softmax(logits, -1)

            # validation output uses teacher forcing
            output = decoder(data, edges, rel_rec, rel_send, 1)

            #target = data[:, :, 1:, :]
            target = data[:, 1:, :, :]
            target = target.transpose(2, 1)
            
            loss_nll = nll_gaussian(output, target, var)
            loss_kl = kl_categorical_uniform(
                prob, num_residues, edge_types)

        acc = edge_accuracy(logits, relations)
        acc_val.append(acc)

        mse_val.append(F.mse_loss(output, target).item())
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())

    print('INFO:: Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(np.array(nll_train))),
          'kl_train: {:.10f}'.format(np.mean(np.array(kl_train))),
          'mse_train: {:.10f}'.format(np.mean(np.array(mse_train))),
          'acc_train: {:.10f}'.format(np.mean(np.array(acc_train))),
          'nll_val: {:.10f}'.format(np.mean(np.array(nll_val))),
          'kl_val: {:.10f}'.format(np.mean(np.array(kl_val))),
          'mse_val: {:.10f}'.format(np.mean(np.array(mse_val))),
          'acc_val: {:.10f}'.format(np.mean(np.array(acc_val))),
          'time: {:.4f}s'.format(time.time() - t))
    edges_train = np.concatenate(edges_train)
    probs_train = np.concatenate(probs_train)
    if np.mean(np.array(nll_val)) < best_val_loss:
        torch.save(encoder.state_dict(), f'./trained/encoder/trained_encoder_{sysname}_{epoch}.pt')
        torch.save(decoder.state_dict(), f'./trained/decoder/trained_decoder_{sysname}_{epoch}.pt')
        torch.save(optimizer.state_dict(), f'./trained/optimizer/trained_optimizer_{sysname}_{epoch}.pt')
        torch.save(scheduler.state_dict(), f'./trained/scheduler/trained_scheduler_{sysname}_{epoch}.pt') 
        print('INFO:: Best model so far, saving...')
        print('INFO:: Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(np.array(nll_train))),
              'kl_train: {:.10f}'.format(np.mean(np.array(kl_train))),
              'mse_train: {:.10f}'.format(np.mean(np.array(mse_train))),
              'acc_train: {:.10f}'.format(np.mean(np.array(acc_train))),
              'nll_val: {:.10f}'.format(np.mean(np.array(nll_val))),
              'kl_val: {:.10f}'.format(np.mean(np.array(kl_val))),
              'mse_val: {:.10f}'.format(np.mean(np.array(mse_val))),
              'acc_val: {:.10f}'.format(np.mean(np.array(acc_val))),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    return encoder, decoder, edges_train, probs_train, nll_val, nll_train, kl_train, mse_train, acc_train, kl_val, mse_val, acc_val

