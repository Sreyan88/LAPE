from src.augmentations.augmentations import *



class AugmentationModule:
    """The Augmentation Module"""

    def __init__(self, config, args, len_of_files):

        self.train_transform = nn.Sequential(
            *self.get_augmentations(config, args, len_of_files)
        )
        if config["pretrain"]["normalization"] == "mean_var":
            self.pre_norm = RunningNorm(epoch_samples= 2*len_of_files) #@Ashish please write rationale why 2*

        print('Augmentations:', self.train_transform)

    def get_augmentations(self, config, args, len_of_files):
        list_augmentations = []

        if "MixupBYOLA" in config["pretrain"]["augmentations"]:
            list_augmentations.append(MixupBYOLA(ratio=config["pretrain"]["augmentations"]["MixupBYOLA"]["ratio"], log_mixup_exp=config["pretrain"]["augmentations"]["MixupBYOLA"]["log_mixup_exp"]))
        if "RandomResizeCrop" in config["pretrain"]["augmentations"]:
            list_augmentations.append(RandomResizeCrop(virtual_crop_scale=config["pretrain"]["augmentations"]["RandomResizeCrop"]["virtual_crop_scale"], freq_scale=config["pretrain"]["augmentations"]["RandomResizeCrop"]["freq_crop_scale"], time_scale=config["pretrain"]["augmentations"]["RandomResizeCrop"]["time_crop_scale"]))
        if "Kmix" in config["pretrain"]["augmentations"]:
            list_augmentations.append(Kmix(args, config, len_of_files, ratio=config["pretrain"]["augmentations"]["Kmix"]["ratio"], log_mixup_exp=config["pretrain"]["augmentations"]["Kmix"]["log_mixup_exp"], top_k=config["pretrain"]["augmentations"]["Kmix"]["top_k"], centroid_path=config["pretrain"]["augmentations"]["Kmix"]["centroid_path"]))
        if "PatchDrop" in config["pretrain"]["augmentations"]:
            list_augmentations.append(PatchDrop(patch_drop=config["pretrain"]["augmentations"]["PatchDrop"]["ratio"]))

        return list_augmentations

    def __call__(self, x):
        if self.pre_norm:
            x = self.pre_norm(x)
        return self.train_transform(x), self.train_transform(x)
