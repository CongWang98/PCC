# -*- coding: utf-8 -*-
# @Time    : 2021/01/24
# @Author  : Cong Wang
# @Github ：https://github.com/CongWang98
import matplotlib
matplotlib.use('agg')
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from sklearn import preprocessing
from model import FCAE, AEparameter
from preprocessing import LoadAngDihFile, SampleData, DivideAdlis
import matplotlib.pyplot as plt
import argparse
import time
from tqdm import tqdm
import warnings


warnings.filterwarnings('ignore')
time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
modelname = 'FCAE'
random_seed = 114514


def split_indices(n, val_pct):
    # Determine size of validation set
    n_val = int(val_pct*n)
    # Create random permutation of 0 to n-1
    np.random.seed(random_seed)
    idxs = np.random.permutation(n)
    # Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]


def get_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b[0], self.device)  # Modified dataloadder

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def PreprocessedDataToLoader(data, batchsize, device, val_pct=0.2, verbose=1):
    x_dataset = TensorDataset(torch.from_numpy(data))
    train_indices, val_indices = split_indices(len(x_dataset), val_pct=0.2)
    train_sampler = SubsetRandomSampler(train_indices)
    train_dl = DataLoader(x_dataset, batchsize, sampler=train_sampler)
    train_dl_for_val = DataLoader(x_dataset, 5000, sampler=train_sampler)
    valid_sampler = SubsetRandomSampler(val_indices)
    valid_dl = DataLoader(x_dataset, 5000, sampler=valid_sampler)
    train_dl = DeviceDataLoader(train_dl, device)
    train_dl_for_val = DeviceDataLoader(train_dl_for_val, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    if verbose:
        print('[INFO] Dataset divded. Train set: {} frames, val set: {} frames.'.format(len(train_indices), len(val_indices)))
    return train_dl, valid_dl, train_dl_for_val


def loss_batch(model, xb, opt=None, useOriginData=False, mean=0, scale=1, lossf=nn.L1Loss()):
    """ Calculate loss on a batch"""
    # Generate predictions
    if useOriginData:
        xb = (xb - mean) / scale
    preds = model(xb)
    if useOriginData:
        xb = mean + xb * scale
        preds = mean + preds * scale
    # Calculate loss
    loss = lossf(xb, preds)
    if opt is not None:
        # Compute gradients
        loss.backward()
        # Update parameters
        opt.step()
        # Reset gradients
        opt.zero_grad()
    return loss.item(), len(xb)


def evaluate(model, valid_dl, useOriginData=False, mean=0, scale=1, lossf=nn.L1Loss()):
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, xb, useOriginData=useOriginData, mean=mean, scale=scale, lossf=lossf)
                   for xb in valid_dl]
        # Separate losses, counts and metrics
        losses, nums = zip(*results)
        # Total size of the dataset
        total = np.sum(nums)
        # Avg. loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
    return avg_loss, total


def fit(epochs, lr, model, data, batchsize, device, val_pct=0.2, opt_fn=torch.optim.Adam, lossf=nn.L1Loss(), output=0, verbose=1):
    # Generate dataloader
    train_dl, valid_dl, train_dl_for_val = PreprocessedDataToLoader(data, batchsize, device, val_pct=0.2, verbose=verbose)
    losses = []
    opt = opt_fn(model.parameters(), lr=lr)
    step_batch = 0
    train_len = len(data) - int(len(data) * val_pct)
    batch_per_epoch = (train_len + batchsize - 1) // batchsize
    for epoch in range(epochs):
        # Training
        if verbose:
            pbar = tqdm(train_dl, leave=False, desc='[INFO] Epoch {}/{}'.format(epoch + 1, epochs))
        else:
            pbar = train_dl
        # for xb in tqdm(train_dl, desc='[{}] Epoch {}/{}'.format(modelname, epoch + 1, epochs), leave=False):
        for xb in pbar:
            step_batch += 1
            loss, _ = loss_batch(model, xb, opt, lossf=lossf)
            if output and (step_batch - batch_per_epoch * epoch) % (batch_per_epoch / 10) == 0:
                writer.add_scalar('batch_loss', loss, global_step=step_batch)
                batch_train_result = evaluate(model, train_dl_for_val, lossf=lossf)
                batch_train_loss, _ = batch_train_result
                batch_val_result = evaluate(model, valid_dl, lossf=lossf)
                batch_val_loss, _ = batch_val_result
                writer.add_scalar('batch_train_loss', batch_train_loss, global_step=step_batch)
                writer.add_scalar('batch_val_loss', batch_val_loss, global_step=step_batch)

        # Evaluation
        train_result = evaluate(model, train_dl_for_val, lossf=lossf)
        train_loss, _ = train_result
        val_result = evaluate(model, valid_dl, lossf=lossf)
        val_loss, _ = val_result
        if output:
            writer.add_scalar('train_loss', train_loss, global_step=epoch + 1)
            writer.add_scalar('val_loss', val_loss, global_step=epoch + 1)
        # Record the loss & metric
        losses.append(val_loss)
        # Print progress
        if verbose:
            print('[INFO] Epoch {}/{}: train_loss: {:.4f}, val_loss: {:.4f}'
                  .format(epoch + 1, epochs, train_loss, val_loss))
        if not os.path.exists('training_result/{}/{}_{}/checkpoint'.format(args.dataset, modelname, time_now)):
            os.makedirs('training_result/{}/{}_{}/checkpoint'.format(args.dataset, modelname, time_now))
        if output and (epoch + 1) % (epochs / 10) == 0:
            torch.save(model.state_dict(), 'training_result/{}/{}_{}/checkpoint/e_{}_valloss_{:.4f}.pth'.format(args.dataset, modelname, time_now, epoch + 1, val_loss))
    if output:
        torch.save(model.state_dict(), 'training_result/{}/{}_{}/checkpoint/final_valloss_{:.4f}.pth'.format(args.dataset, modelname, time_now, val_loss))
    return losses


def GenLatentFile(model, filepath, folderpath, dev, batchsize=1000):
    '''
    Generate a file contain latent space.
    '''
    _, _, adlis = LoadAngDihFile(filepath, verbose=0)
    # print(adlis)
    adstandlis = (adlis - stand_scaler.mean_)/stand_scaler.scale_
    adset = TensorDataset(torch.tensor(adstandlis, dtype=torch.float32))
    x_dl = DataLoader(adset, batchsize)
    x_dl = DeviceDataLoader(x_dl, dev)
    ifilename = re.split(r'[/\\]', filepath)[-1]
    ofilename = re.split(r'[.]', ifilename)[0] + '.latent'
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    with open(folderpath + '/' + ofilename, 'w') as f:
        for xb in tqdm(x_dl, desc='[INFO] {}'.format(ofilename), leave=False):
            model(xb)
            latentb = model.z_mean
            for line in latentb:
                f.write(' '.join([str(latent_ele.item()) for latent_ele in line]) + '\n')


def GenAllLatentFile(model, path, outfolder, dev):
    '''
    Generate latent file for all angdih files in a folder.
    '''
    files = os.listdir(path)
    files.sort()
    count = 0
    for file in tqdm(files, desc='[INFO] {}'.format(path)):
        if file.split('.')[-1] == 'angdih':
            GenLatentFile(model, path + '/' + file, outfolder, dev)
            count += 1
    print('[INFO] {} latent files from {} generated'.format(count, path))


def log():
    log_file = args.dataset + '.log'
    with open(log_file, 'a') as f:
        f.write('time: {}\tmodel: {}\tdataset: {}\t samplerate: {}\tbatchsize: {}\tepoch: {} \tlr: {}\topt: {}\ttime_cost: {:.2f} s\n'
                .format(time_now, modelname, DATASET, SAMPLE_RATE, BATCH_SIZE, EPOCH, LR, OPT.__name__, tcost))
        f.write('latent space dimension: {}\tinter_dims:{}\n'.format(LATENT_DIM, INTER_DIMS))
        f.write('normalized_total_loss: {}\t, total_loss:{}\n'.format(adtotal_loss, total_loss))

    print('[INFO]] Log file has been created.')


def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('-ds', '--dataset', default='example')
    ap.add_argument('-bs', '--batchsize', default=1000, type=int)
    ap.add_argument('-ep', '--epoch', default=100, type=int)
    ap.add_argument('-opt', '--optimizefunction', default='Adam', type=str)
    ap.add_argument('-lr', '--learningrate', default=0.001, type=float)
    ap.add_argument('-no', '--no_output', action='store_false')
    ap.add_argument('-sr', '--samplerate', default=0.1, type=float)
    ap.add_argument('-ld', '--latentdim', default=10, type=int)
    ap.add_argument('-ids', '--inter_dims', nargs='+', type=int, default=[1000, 1000, 1000])
    return ap.parse_args()


if __name__ == "__main__":
    args = args_parse()
    DATASET = 'dataset/' + args.dataset
    BATCH_SIZE = args.batchsize
    EPOCH = args.epoch
    LR = args.learningrate
    if args.optimizefunction == 'Adam':
        OPT = torch.optim.Adam
    elif args.optimizefunction == 'SGD':
        OPT = torch.optim.SGD
    else:
        raise ValueError('Optim function not supported now')
    SAMPLE_RATE = args.samplerate

    OUTPUT = args.no_output
    LATENT_DIM = args.latentdim
    INTER_DIMS = args.inter_dims

    print('[INFO] batch size: {}, epoch: {}'.format(BATCH_SIZE, EPOCH))
    print('[INFO] optim function: {}, learning rate: {}'.format(OPT.__name__, LR))
    device = get_device()
    print('[INFO] device: {}'.format(device))

    # Extract dataset from a give folder
    filelis = os.listdir(DATASET)
    filelis.sort()
    x_lis = []
    atomset = set()
    frametotal = 0
    filecount = 0
    for file in tqdm(filelis, desc='[INFO] Loading trajectory in {}'.format(DATASET), leave=False):
        if file.split('.')[-1] != 'angdih':
            continue
        filecount += 1
        frame, atom, tmp = LoadAngDihFile(DATASET + '/' + file, verbose=0)
        frametotal += frame
        atomset.add(atom)
        # tmplis = SampleData(tmp, SAMPLE_RATE)
        x_lis.append(tmp)
    print('[INFO] {} trajectory loaded in {}'.format(filecount, DATASET))
    xtotal = x_lis[0]
    for i in range(len(x_lis) - 1):
        xtotal = np.vstack((xtotal, x_lis[i + 1]))

    # preprocess
    stand_scaler = preprocessing.StandardScaler()
    xtotal_stand = stand_scaler.fit_transform(xtotal)
    xmean = to_device(torch.tensor(stand_scaler.mean_, dtype=torch.float32), device)
    xscale = to_device(torch.tensor(stand_scaler.scale_, dtype=torch.float32), device)
    print('xmean:', xmean)
    print('xscale:', xscale)
    x_stand = SampleData(xtotal_stand, SAMPLE_RATE)
    if len(atomset) != 1:
        raise ValueError('wrong atom number.')
    DATASET_SIZE = len(x_stand)
    ATOM_NUM = atomset.pop()
    print('[INFO] All files loaded. Frame in total: {}, atom: {}, frame sampled: {}'
          .format(frametotal, ATOM_NUM, DATASET_SIZE))

    # Construct the model
    param = AEparameter(2 * ATOM_NUM - 5, INTER_DIMS, LATENT_DIM)
    net = FCAE(param)
    to_device(net, device)

    if OUTPUT:
        writer = SummaryWriter(logdir='training_result/{}/{}_{}/tensorboard'.format(args.dataset, modelname, time_now))

    # Print model's state_dict
    print("[INFO] Model's stucture:")
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
    # Train the model
    tbegin = time.time()
    losses1 = fit(EPOCH, LR, net, x_stand, BATCH_SIZE, device, opt_fn=OPT, output=OUTPUT, verbose=1)
    tend = time.time()
    tcost = tend - tbegin
    print('[INFO] time cost: {:.2f} s'.format(tcost))

    # Convert np.array to torch Dataloader
    xtotal_dataset = TensorDataset(torch.from_numpy(xtotal_stand))
    adtotal_dataset = TensorDataset(torch.from_numpy(xtotal))
    xtotal_dl = DataLoader(xtotal_dataset, 5000)
    adtotal_dl = DataLoader(adtotal_dataset, 5000)

    # Warp the dataloader to the given device.
    xtotal_dl = DeviceDataLoader(xtotal_dl, device)
    adtotal_dl = DeviceDataLoader(adtotal_dl, device)

    # Print the performance on the whole dataset
    total_loss, _ = evaluate(net, xtotal_dl, lossf=nn.L1Loss())
    adtotal_loss, _ = evaluate(net, adtotal_dl, useOriginData=True,
                               mean=xmean, scale=xscale, lossf=nn.L1Loss())
    print('[INFO] loss on whole set: {:.4f}, origin loss: {:.4f}'
          .format(total_loss, adtotal_loss))

    params = list(net.named_parameters())
    print(params.__len__())
    print(params[0])
    print(params[-1])

    # Generate latent file
    if OUTPUT:
        latent_path = 'training_result/{}/{}_{}/latent'.format(args.dataset, modelname, time_now)
        os.makedirs(latent_path)
        GenAllLatentFile(net, DATASET, latent_path, device)

    # Write the log file
    if OUTPUT:
        log()

    # Output the comparison images
    if OUTPUT:
        pre_tmplist = []
        for dl in tqdm(xtotal_dl, desc='[INFO] predicting...'):
            tmp = net(dl)
            pre_tmplist.append(np.array((tmp * xscale + xmean).cpu().detach()))
        preadlis = pre_tmplist[0]
        for i in range(len(pre_tmplist) - 1):
            preadlis = np.vstack((preadlis, pre_tmplist[i + 1]))
        addelis = DivideAdlis(xtotal)
        preaddelis = DivideAdlis(preadlis)
        outpath = 'training_result/{}/{}_{}/image'.format(args.dataset, modelname, time_now)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        fig = plt.figure(figsize=(10, 5))
        for n in tqdm(range(ATOM_NUM * 2 - 5), desc='[INFO] plotting...'):
            if n % 2:
                title = 'Distribution of dihedral {}'.format(1 + n // 2)
                xlable = r'dihedral angle (unit: ${\rm 2\pi}$)'
                outfile = 'dihedral_{}.png'.format(1 + n // 2)
            else:
                title = 'Distribution of angle {}'.format(1 + n // 2)
                xlable = r'angle (unit: ${\rm \pi}$)'
                outfile = 'angle_{}.png'.format(1 + n // 2)
            ax1 = plt.subplot(1, 2, 1)
            range_ = (min(addelis[n]), max(addelis[n]))
            plt.hist(addelis[n], bins=100, range=range_)
            plt.title('input')
            plt.xlabel(xlable)
            plt.ylabel('frequency')
            # ax1.set_title('Input')
            ax2 = plt.subplot(1, 2, 2)
            plt.hist(preaddelis[n], bins=100, range=range_)
            plt.title('prediction')
            plt.xlabel(xlable)
            plt.ylabel('frequency')
            fig.tight_layout()
            plt.suptitle(title)
            plt.savefig(outpath + '/' + outfile)
            plt.clf()
        plt.close()
        print('[INFO] {} images have been plotted.'.format(ATOM_NUM * 2 - 5))
