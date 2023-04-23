import os
import time
import json
import sys
import yaml
import importlib
import argparse
import torch
import logging
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
from src.augmentations import AugmentationModule
from src.utils import check_downstream_hf_availability
from src.downstream.downstream_encoder import DownstreamEncoder
from src.dataset.downstream_dataset import DownstreamDataset,DownstreamDatasetHF
from src.utils import freeze_encoder, get_logger, AverageMeter, Metric, load_pretrained_encoder

def main(gpu, args):

    if args.config is None:
        default_downstream_config = "src/downstream/downstream_config.yaml"
        with open(default_downstream_config, 'r') as duc:
            config = yaml.load(duc, Loader=yaml.FullLoader)
    else:
        with open(args.config, 'r') as duc:
            config = yaml.load(duc, Loader=yaml.FullLoader)
    print(config)

    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    stats_file=None
    args.exp_root = args.exp_dir / args.task
    args.exp_root.mkdir(parents=True, exist_ok=True)
    
    if args.rank == 0:
        stats_file = open(args.exp_root / 'downstream_stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    logger = get_logger(args)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True # ! change it set seed

    
    assert config['run']['batch_size'] % args.world_size == 0
    per_device_batch_size = config['run']['batch_size'] // args.world_size


    eval_dataset = None
    eval_loader = None
    # If the dataset is availble in HuggingFace
    if check_downstream_hf_availability(args.task) == "hf":
        train_dataset = DownstreamDatasetHF(args,config,split='train')
        test_dataset = DownstreamDatasetHF(args,config,split='test')
        if config['run']['eval']:
            eval_dataset = DownstreamDatasetHF(args,config,split='validation')
    # If the dataset is NOT availble in HuggingFace
    else:
        train_dataset = DownstreamDataset(args,config,split='train')
        test_dataset = DownstreamDataset(args,config,split='test',labels_dict=train_dataset.labels_dict)
        if config['run']['eval']:
            if args.valid_csv:
                eval_dataset = DownstreamDataset(args,config,split='validation',labels_dict=train_dataset.labels_dict)
            else:
                raise Exception('Evaluation will be done since eval=True set in config but no validation csv specified.')
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=1) #shuffle
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=per_device_batch_size,
                                                pin_memory=True,sampler = train_sampler,num_workers=0)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, seed=1) #shuffle
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=per_device_batch_size,
                                                pin_memory=True, num_workers=0)

    if eval_dataset is not None:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, seed=1) #shuffle
        eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=per_device_batch_size,
                                                pin_memory=True,sampler = eval_sampler,num_workers=0)

    # override the encoder if encoder is specified
    if args.encoder is not None:
        config['downstream']['base_encoder']['type'] = args.encoder

    #load base encoder
    module_path_base_encoder = f'src.encoder'
    base_encoder = getattr(importlib.import_module(module_path_base_encoder), config["downstream"]["base_encoder"]["type"])
    model = DownstreamEncoder(config, args, base_encoder, no_of_classes=train_dataset.no_of_classes).cuda(gpu) 
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.freeze:
        freeze_encoder(model)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if args.checkpoint is not None:
        # Working need to make it work for ddp pretraining
        load_pretrained_encoder(model,args)
    
    
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=config['run']['lr'],
    )

    if args.rank == 0 : logger.info("started training")

    train_accuracy=[]
    train_losses=[]
    eval_accuracy=[]
    eval_losses=[]
    test_accuracy=[]
    test_losses=[]

    for epoch in range(0, config["run"]["epochs"]):
        train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, epoch,gpu,args)

        if eval_loader is not None:
            if args.rank == 0 :
                eval_stats = eval(epoch,model,eval_loader,criterion,gpu)
                eval_accuracy.append(eval_stats["accuracy"].avg)
                stats = dict(epoch=epoch,
                        Train_loss=eval_stats["loss"].avg.cpu().numpy().item(),
                        Test_Loss=(eval_stats["loss"].avg).numpy().item(),
                        Test_Accuracy =eval_stats["accuracy"].avg,
                        Best_Test_Acc=max(eval_accuracy))
                print(stats)
                print(json.dumps(stats), file=stats_file)
        
        if ((epoch + 1) % config['run']['test_every_n_epochs']) == 0:
            if args.rank == 0 :
                test_stats = eval(epoch,model,test_loader,criterion,gpu)
                test_accuracy.append(eval_stats["accuracy"].avg)
                stats = dict(epoch=epoch,
                        Train_loss=train_stats["loss"].avg.cpu().numpy().item(),
                        Test_Loss=(test_stats["loss"].avg).numpy().item(),
                        Test_Accuracy =test_stats["accuracy"].avg,
                        Best_Test_Acc=max(test_accuracy))
                print(stats)
                print(json.dumps(stats), file=stats_file)
    
    if args.rank ==0 :
        print("max validation accuracy : {}".format(max(eval_accuracy)))
        print("max test accuracy : {}".format(max(test_accuracy)))
        plt.plot(range(1,len(eval_accuracy)+1), eval_accuracy, label = "train accuracy",marker = 'x')
        plt.legend()
        plt.savefig(args.exp_root / 'accuracy.png')


def train_one_epoch(loader, model, crit, opt, epoch,gpu,args):
    '''
    Train one Epoch
    '''
    logger = logging.getLogger(__name__)
    logger.debug("epoch:"+str(epoch) +" Started")
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    model.train() # ! imp
    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        output = model(input_tensor.float().to(gpu))
        loss = crit(output, target.to(gpu))

        losses.update(loss.data, input_tensor.size(0))
        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 :
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                    .format(epoch, i, len(loader), batch_time=batch_time,
                            data_time=data_time, loss=losses))


    logger.debug("epoch-"+str(epoch) +" ended")
    stats = dict(epoch=epoch,loss=losses)
    return stats

@torch.no_grad()
def eval(epoch,model,loader,crit,gpu):
    model.eval() # ! Imp
    losses = AverageMeter()
    accuracy = Metric()
    with torch.no_grad():
        for step, (input_tensor, targets) in enumerate(loader):
           # input_tensor = torch.squeeze(input_tensor,0)
            if torch.cuda.is_available():
                input_tensor =input_tensor.cuda(gpu ,non_blocking=True)
                targets = targets.cuda(gpu,non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(input_tensor.float())
                loss = crit(outputs, targets)
                preds = torch.argmax(outputs,dim=1)==targets

            accuracy.update(preds.cpu())
            losses.update(loss.cpu().data, input_tensor.size(0))

    stats = dict(epoch=epoch,loss=losses, accuracy = accuracy)
    return stats

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--task", help="path to data directory", type=str, default='speech_commands_v1')
    parser.add_argument("--train_csv", help="path to data directory", type=str, default='/speech/ashish/test_label_data.csv')
    parser.add_argument("--valid_csv", help="path to data directory", type=str, default=None)
    parser.add_argument("--test_csv", help="path to data directory", type=str, default='/speech/ashish/test_label_data.csv')
    parser.add_argument('--checkpoint', type=str, help='path to pre-trained checkpoint', default = None)
    parser.add_argument('--encoder', type=str, help='type of encoder you want to use', default = 'AudioNTT2020Task6')
    parser.add_argument('--freeze', type=bool, help='if you want to freeze the encoder for downstream fine-tuning', default = False)
    parser.add_argument('--exp_dir',default='./exp',type=Path,help="experiment root directory")
    parser.add_argument('--upstream', type=str, help='define the type of upstream', default = 'delores_m')
    parser.add_argument('-c', '--config', metavar='CONFIG_PATH', help='The yaml file for configuring the whole experiment, except the upstream model', default = "src/downstream/downstream_config.yaml")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = 'tcp://localhost:58367'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main, (args,), args.ngpus_per_node)
