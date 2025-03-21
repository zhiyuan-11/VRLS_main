"""Based on https://github.com/MasaKat0/D3RE/
"""
import sys
sys.path.append('..')

# from labelshift import *
# from utils4gluon import *
# from MLLS import *

import time
from numpy import math

import torch
import torch.optim as optim

from logger import getLogger


def run(net, temp_calibration, Xtest):
    
    ypred_EM_preact = predict_all_preact(Xtest, net, ctx=mx.cpu(), dfeat=784)

    ypred_EM_calib_soft = temp_calibration(ypred_EM_preact)
    ypred_EM_calib = np.argmax(ypred_EM_calib_soft,axis = 1)

    Py_EM = calculate_marginal(ypred_EM_calib,10) # ypred_val_calib
    Py_EM = Py_EM.reshape((10))

    wt_em = EM(Py_EM, ypred_EM_calib_soft, 10) # /Py_base
    
    return wt_em # , Py_base
    
def run_resnet(net, temp_calibration, Xtest, Py_base):
    

    samples = Xtest.shape[0]
    batch_size = 200
    num_batch = samples//batch_size
    
    ypred_test_preact = np.zeros((samples, 10))

            
    with torch.no_grad():
        for j in range(num_batch):
            start_idx = j * batch_size
            end_idx = (j + 1) * batch_size
            ypred_test_preact[start_idx:end_idx, :] = net(Xtest[start_idx:end_idx].float()).cpu().numpy()

        # Handle any remaining samples that don't fit into a full batch
        if samples % batch_size != 0:
            start_idx = num_batch * batch_size
            ypred_test_preact[start_idx:, :] = net(Xtest[start_idx:].float()).cpu().numpy()
    

    ypred_test_calib_soft = temp_calibration(ypred_test_preact)

    wt_em = EM(Py_base, ypred_test_calib_soft, 10) # /Py_base
    
    return wt_em # , Py_base
    
            

class LS_RatioModel(torch.nn.Module):
    def __init__(self, net, temp_calibration):
        super().__init__()
        self.net = net
        self.temp_calibration = temp_calibration
    def forward(self, Xtest):
        wt = run(self.net, self.temp_calibration, Xtest) 
        return torch.tensor(wt)
        
class LS_RatioModel_resnet(torch.nn.Module):
    def __init__(self, net, temp_calibration):
        super().__init__()
        self.net = net
        self.temp_calibration = temp_calibration
    def forward(self, Xtest, Py_base):
        wt = run_resnet(self.net, self.temp_calibration, Xtest, Py_base) 
        return torch.tensor(wt)
        
class TrueRatioModel(torch.nn.Module):
    def __init__(self, ratio_target_lookup):
        super().__init__()
        self.ratio_target_lookup = ratio_target_lookup
        self.use_target = True

    def forward(self, data, targets):
        return self.ratio_target_lookup[targets].to(data.device)


class UniformRatioModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor(1.0).repeat(x.shape[0]).to(x.device)


class ForcePositive(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.eps = torch.tensor(0.0001)

    def forward(self, x):
        o = self.model(x)
        #return torch.sign(o)*o
        return torch.max(o, self.eps)


class RatioEstimation(object):
    def __init__(self, id_, ratio_model, trainset, public_testset, args):
        self.id_ = id_
        self.ratio_model = ratio_model
        self.trainset = trainset
        self.public_testset = public_testset
        self.args = args
        self.make_dataloaders(self.args.re_batch_size)

    def make_dataloaders(self, batchsize):
        drop_last_train = self.args.re_batch_drop_last and batchsize < len(self.trainset)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, 
            batch_size=batchsize, 
            shuffle=True, 
            drop_last=drop_last_train, 
            **self.args.dataloader_kwargs)
        #factor = len(public_testset)/len(trainset)
        #self.public_test_loader = torch.utils.data.DataLoader(public_testset, batch_size=int(math.ceil(factor * batchsize)), shuffle=True, **self.args.dataloader_kwargs)
        drop_last_test = self.args.re_batch_drop_last and batchsize < len(self.public_testset)
        self.public_test_loader = torch.utils.data.DataLoader(self.public_testset, 
            batch_size=batchsize, 
            shuffle=True,
            drop_last=drop_last_test, 
            **self.args.dataloader_kwargs)

    def train(self, on_epoch_end=None, log=True):
        assert self.args.re_type == 'lsif', "Only LSIF supported"
        logger = getLogger(__name__)

        # Set optimizer
        if self.args.re_optimizer == 'adam':
            optimizer = optim.Adam(self.ratio_model.parameters(), 
                lr=self.args.re_lr, 
                weight_decay=self.args.re_weight_decay)
        elif self.args.re_optimizer == 'sgd':
            optimizer = optim.SGD(self.ratio_model.parameters(), 
                lr=self.args.re_lr, 
                momentum=self.args.re_momentum,
                weight_decay=self.args.re_weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=self.args.re_lr_milestones, 
            gamma=0.1)

        # Training
        logger.info('Starting ratio estimation training...')
        self.ratio_model.train()
        start_time = time.time()
        for epoch in range(self.args.re_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            
            public_test_loader_iterator = iter(self.public_test_loader)
            for train_data in self.train_loader:
                # To loop over two dataloaders simultaneously: 
                # https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch
                try:
                    test_data = next(public_test_loader_iterator)
                except StopIteration:
                    public_test_loader_iterator = iter(self.public_test_loader)
                    test_data = next(public_test_loader_iterator)

                train_inputs = train_data[0].to(self.args.device)
                test_inputs = test_data[0].to(self.args.device)
                train_targets = train_data[1].to(self.args.device)
                test_targets = test_data[1].to(self.args.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                if self.args.model == 'label-based':
                    train_outputs = torch.t(self.ratio_model(train_targets))[0]
                    test_outputs = torch.t(self.ratio_model(test_targets))[0]
                else:
                    train_outputs = torch.t(self.ratio_model(train_inputs))[0]
                    test_outputs = torch.t(self.ratio_model(test_inputs))[0]

                # l1 = lambda z: z**2
                # l2 = lambda z: -(2*z-z**2/self.args.re_upper_bound)
                # tr1 = l1(train_outputs).mean()
                # te1 = l1(test_outputs).mean()/self.args.re_upper_bound
                # te2 = l2(test_outputs).mean()

                # if tr1 - te2 >= 0:
                #     loss = tr1 - te1 + te2
                # else:
                #     loss = - (tr1 - te1)

                n_tr, n_te = train_inputs.shape[0], test_inputs.shape[0]

                loss_te = ((-test_outputs+test_outputs**2/(2*self.args.re_upper_bound))).sum()/n_te
                loss_te_middle = (test_outputs**2/(2*self.args.re_upper_bound)).sum()/n_te
                loss_tr = (train_outputs**2/2).sum()/n_tr

                if not self.args.re_descent_only and loss_tr - loss_te_middle < 0:
                    loss = - (loss_tr - loss_te_middle)
                else:
                    loss = loss_te - loss_te_middle + loss_tr
                
                loss.backward()
                optimizer.step()

                # logger.info(f'  Epoch {epoch}: {n_batches} / {len(self.train_loader)} batches')

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss = epoch_loss / n_batches

            # log epoch statistics
            if log:
                epoch_train_time = time.time() - epoch_start_time
                logger.info(f'| Epoch: {epoch + 1:03}/{self.args.re_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                            f'| Train Loss: {epoch_loss:.6f} |')
            
            if on_epoch_end is not None:
                on_epoch_end({'epoch': epoch, 'train_loss': epoch_loss, 'lr': scheduler.get_lr()[0]})

            # Step scheduler
            scheduler.step()
            if epoch in self.args.re_lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            # Batch scheduler
            if self.args.re_batch_size_snd_epoch is not None and self.args.re_batch_size_snd_epoch == epoch:
                logger.info(f'Changes batchsize to {self.args.re_batch_size_snd}')
                self.make_dataloaders(self.args.re_batch_size_snd)

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')
