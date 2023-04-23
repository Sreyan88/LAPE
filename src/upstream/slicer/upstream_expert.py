import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from typing import Union
#from pl_bolts.metrics import mean, precision_at_k
# from torchmetrics import Precision

from src.utils import off_diagonal, concat_all_gather, ClusterLoss
from src.upstream.slicer.upstream_encoder import SLICER as SLICER_ENCODER

class Upstream_Expert(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Moco <https://arxiv.org/abs/2003.04297>`_
    Paper authors: Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He.
    Code adapted from `facebookresearch/moco <https://github.com/facebookresearch/moco>`_ to Lightning by:
        - `William Falcon <https://github.com/williamFalcon>`_
    Example::
        from pl_bolts.models.self_supervised import Moco_v2
        model = Moco_v2()
        trainer = Trainer()
        trainer.fit(model)
    CLI command::
        # cifar10
        python moco2_module.py --gpus 1
        # imagenet
        python moco2_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32
    """

    def __init__(
        self,
        config,
        base_encoder,
        emb_dim: int = 128,
        num_negatives: int = 65536,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.07,
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        data_dir: str = './',
        batch_size: int = 256,
        use_mlp: bool = False,
        num_workers: int = 8,
        *args,
        **kwargs
    ):
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            data_dir: the directory to store data
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.base_encoder = base_encoder
  
        self.encoder_q, self.encoder_k = self.init_encoders(self.base_encoder)

        if use_mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        # copy weights between student and teacher
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.loss_cluster = ClusterLoss(self.config["pretrain"]["cluster_contrastive_dim"], 1, None)
        

    def init_encoders(self, base_encoder):
        """
        Override to add your own encoders
        """
        encoder_q = SLICER_ENCODER(self.config, base_encoder)
        encoder_k = SLICER_ENCODER(self.config, base_encoder)

        return encoder_q, encoder_k
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.trainer.strategy =="ddp":
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no cover
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no cover
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, img_q=None, img_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q_instance, q_cluster = self.encoder_q(img_q)  # queries: NxC
        q_instance = nn.functional.normalize(q_instance, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            if self.trainer.strategy =="ddp":
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)
            k_instance, k_cluster = self.encoder_k(img_k)  # keys: NxC
            k_instance = nn.functional.normalize(k_instance, dim=1)
            # undo shuffle
            if self.trainer.strategy =="ddp":
                k_instance = self._batch_unshuffle_ddp(k_instance, idx_unshuffle)
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_instance, k_instance]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_instance, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.hparams.softmax_temperature
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_instance)

        return logits, labels, q_cluster, k_cluster 

    def training_step(self, batch, batch_idx):
        
        img_1, img_2 = batch
        output, target, q_cluster, k_cluster = self(img_q=img_1, img_k=img_2)
        output_1, target_1, q_cluster_1, k_cluster_1 = self(img_q=img_2, img_k=img_1)

        loss = F.cross_entropy(output.float(), target.long())
        loss_1 = F.cross_entropy(output_1.float(), target_1.long())
        sym_loss_instance = loss + loss_1
        loss_cluster = self.loss_cluster(q_cluster, q_cluster_1)
        loss_combine = sym_loss_instance + loss_cluster
        log = {'train_loss': loss_combine, 'sym_instance_loss': sym_loss_instance, 'train_loss_cluster':loss_cluster}
        print(log)
        self.log_dict(log)

        return loss

    def validation_step(self, batch, batch_idx):
        # in STL10 we pass in both lab+unl for online ft
        if self.trainer.datamodule.name == 'stl10':
            # labeled_batch = batch[1]
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (img_1, img_2), labels = batch

        output, target,q1,q2,q3,q4,k1,k2,k3,k4 = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())

        loss+= self.p1(q1,k1)
        loss+= self.p2(q2,k2)
        loss+= self.p3(q3,k3)
        loss+= self.p4(q4,k4)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {'val_loss': val_loss, 'val_acc1': val_acc1, 'val_acc5': val_acc5}
        self.log_dict(log)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
