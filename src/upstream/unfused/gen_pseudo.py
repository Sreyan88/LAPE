import numpy as np
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from src.upstream.unfused.clustering import Kmeans
from src.dataset import BaseDataset
import pandas as pd
import time

AUDIO_SR = 16000

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_features(args, config, dataloader, model, N): #N is total dataset size
    batch = config["run"]["batch_size"]
    verbose = True
    if verbose:
        print('Compute features')
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            aux = model(input_var)[0].data.cpu().numpy() #feature from the final layer

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
                # special treatment for final batch
                features[i * batch:] = aux
                        
    return features



def make_dataset(audio_files, audio_labels, audio_indexes):
    label_to_idx = {label: idx for idx, label in enumerate(set(audio_labels))}
    audiopath_w_labels = []
    for i, index in enumerate(audio_indexes):
        path = audio_files[index]
        pseudolabel = label_to_idx[audio_labels[index]] #could have been pseudolabels, bekar confusion change later
        audiopath_w_labels.append((path,pseudolabel))
            
    return audiopath_w_labels



def gen_pseudo_label(gpu, config, args, base_encoder):
    args.rank += gpu
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    np.random.seed(31)
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    
    list_of_files_directorys = pd.read_csv(args.input)
    list_of_files_directory = list(list_of_files_directorys["files"])
    
    final_model = base_encoder(config["pretrain"]["input"]["n_mels"], config["pretrain"]["base_encoder"]["output_dim"], config["pretrain"]["base_encoder"]["return_all_layers"]).cuda(gpu)
    final_model = nn.SyncBatchNorm.convert_sync_batchnorm(final_model)
    final_model = nn.parallel.DistributedDataParallel(final_model, device_ids=[gpu],find_unused_parameters=True)
    
    cudnn.benchmark = True
    checkpoint = torch.load(config["pretrain"]["pseudo_label_generation"]["teacher_model_ckpt"])
    new_state_dict = {'module'+'.'+'.'.join(key.split('.')[2:]):value for key, value in checkpoint['state_dict'].items()}
    final_model.load_state_dict(new_state_dict, strict=False)    
    pretrain_dataset = BaseDataset(config, args, args.input, None, None)  #without augmentation 

    train_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=config["run"]["batch_size"], num_workers=config["run"]["num_dataloader_workers"])

    features = compute_features(args, config, train_loader, final_model, len(list_of_files_directory))
    deepcluster = Kmeans(config["pretrain"]["pseudo_label_generation"]["labels"])
    clustering_loss = deepcluster.cluster(features, verbose=True)
    
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(deepcluster.images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    indexes_sorted = np.argsort(image_indexes)  
    pseudolabels = np.asarray(pseudolabels)[indexes_sorted]
    dataset_w_labels = make_dataset(list_of_files_directory,pseudolabels,indexes_sorted)

    with open(args.input.split('.')[0]+'_new'+'.'+args.input.split('.')[1],'w') as f:
        f.write('files'+','+'labels'+'\n')
        for x in dataset_w_labels:
            f.write(x[0]+','+str(x[1])+'\n')

def get_pseudo_label(config, args, base_encoder):
    
    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = config["run"]["world_size"]

    torch.multiprocessing.spawn(gen_pseudo_label, (config, args, base_encoder), args.world_size)            
