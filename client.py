import sys
sys.path.append('..')
import torch
import wandb
import torchvision

from args import Args
from ratio_estimation import RatioEstimation
from target_shift import CombinedInMemoryDataset
from ratio_estimation_d3re import RatioEstimation as D3RERatioEstimation
from utils import AverageMeter

    


class Client(object):
    def __init__(self, id_, trainset, testset, model, ratio_model, criterion, args):
        self.id_ = id_
        self.criterion = criterion
        self.args: Args = args

        testset_size = len(testset)
        assert args.public_testset_size < testset_size
        public_testset, testset = torch.utils.data.random_split(testset, [args.public_testset_size, testset_size - args.public_testset_size])

        assert isinstance(args.dataloader_kwargs, dict)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, **self.args.dataloader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.args.test_batch_size, shuffle=False, **self.args.dataloader_kwargs)
        
        self.public_testset = public_testset
        self.trainset = trainset
        self.testset = testset

        self.ratio_model = ratio_model
        self.model = model

    def loss(self, output, target, reduction="mean"):
        return self.criterion(output, target, reduction=reduction)

    def pred(self, output):
        return output.max(1, keepdim=True)[1]

    def get_public_testset(self):
        return self.public_testset

    def estimate_ratio(self, public_testset=None, on_epoch_end=None):
        # TODO: technically the shuffling should not happen on the client
        # TODO: and labels should not be available
        if public_testset is None:
            public_testset = self.public_testset

        if self.args.d3re_impl:
            dataset = CombinedInMemoryDataset(self.trainset, self.public_testset)
            re = D3RERatioEstimation(self.id_, self.ratio_model, dataset, self.args)
        else:
            re = RatioEstimation(self.id_, self.ratio_model, self.trainset, public_testset, self.args)

        re.train(on_epoch_end=on_epoch_end)
    
    # def train_iterator(self):
    #     """Trains the model and computes gradients without zeroing them."""
    #     self.loss_avg = AverageMeter()
    #     self.acc_avg = AverageMeter()
    #     self.model.train()
    #     epoch = 0
        
    #     while True:
    #         # Reset average meters for each epoch
    #         self.loss_avg.reset()
    #         self.acc_avg.reset()
            
    #         for batch_idx, (data, target) in enumerate(self.train_loader):
    #             data, target = data.to(self.args.device), target.to(self.args.device)
    #             batch_size = data.shape[0]
        
    #             # Initialize gradients
    #             self.model.zero_grad()
                
    #             # Forward pass and obtaining features
    #             out_of_local = self.model(data)
    #             local_features = out_of_local[:-1]  # Branch outputs as features
    #             log_probs = out_of_local[-1]        # Final output probabilities (logits)
                
    #             # Lists for branch-wise losses
    #             log_prob_branch = []
    #             ce_branch = []
    #             kl_branch = []
    #             num_branch = len(local_features)
        
    #             # Calculate branch losses
    #             for it in range(num_branch):
        
    #                 # Predict on branch output using global model
    #                 this_log_prob = self.model(local_features[it])
                    
    #                 # Cross-Entropy (CE) and Knowledge Distillation (KD) losses for branch
    #                 this_ce = self.loss(this_log_prob, target, reduction="none")
    #                 this_kl = KD(this_log_prob, log_probs, self.args.temp)
                    
    #                 # Store branch losses
    #                 log_prob_branch.append(this_log_prob)
    #                 ce_branch.append(this_ce.mean())
    #                 kl_branch.append(this_kl.mean())
        
    #             # Final output CE loss
    #             ce_loss = self.loss(log_probs, target, reduction="none")
                
    #             # Combined loss with lambda hyperparameters
    #             total_loss = (ce_loss.mean() + 
    #                           sum(ce_branch) / num_branch + 
    #                           sum(kl_branch) / num_branch)
                
    #             # Backward pass
    #             total_loss.backward()
                
    #             # Update average loss and accuracy
    #             self.loss_avg.update(total_loss.item())
    #             pred = self.pred(log_probs)
    #             self.acc_avg.update((pred == target.view_as(pred)).float().mean().item())
    
    #             # Yield the total loss
    #             yield total_loss
    
    #         # Log epoch metrics with wandb
    #         wandb.log({
    #             f'client_{self.id_}_train_loss': self.loss_avg.avg, 
    #             f'client_{self.id_}_train_acc': self.acc_avg.avg, 
    #             f'client_{self.id_}_epoch': epoch,
    #         })
    #         epoch += 1

    def train_iterator(self):
        """Promises to compute gradients for the model and not zero them.
        """
        self.loss_avg = AverageMeter()
        self.acc_avg = AverageMeter()
        self.model.train()
        epoch = 0
        while True:
            self.loss_avg.reset()
            self.acc_avg.reset()
            
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                batch_size = data.shape[0]
                
                # if batch_idx == 0:
                #     self._log_examples(data)

                self.model.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target, reduction="none")

                # weight data examples
                if hasattr(self.ratio_model, 'use_target') and self.ratio_model.use_target:
                    ratio = self.ratio_model(data, target)
                else:
                    ratio = self.ratio_model(data)
                ratio = ratio.flatten()
                assert ratio.shape == loss.shape
                loss = torch.sum(ratio * loss) / batch_size 
                
                loss.backward()
                self.loss_avg.update(loss.item())
                pred = self.pred(output)
                self.acc_avg.update(torch.sum(pred == target.view_as(pred)) / batch_size)

                yield loss
            wandb.log({
                f'client_{self.id_}_train_loss': self.loss_avg.avg, 
                f'client_{self.id_}_train_acc': self.acc_avg.avg, 
                f'client_{self.id_}_epoch': epoch,
            })
            epoch += 1

    def _log_examples(self, img, N=20):
        # For debugging possible augmentations and the attack
        img_sample = img[:N]
        grid = torchvision.utils.make_grid(img_sample)
        wandb.log({f'client_{self.id_}_train_img': wandb.Image(grid)}, commit=False)
