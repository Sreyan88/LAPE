# LAPE
We introduce LAPE, a unified framework for **L**ow-resource **A**udio **P**rocessing and **E**valuation.

## What is LAPE?
LAPE is an easy-to-use toolkit for audio processing. In its initial release, LAPE supports Self-Supervised Learning (SSL)-based Upstream Pre-training and Downstream Fine-tuning. LAPE, originally introduced in this paper, integrates all our research on low-resource audio processing in one unified framework. We open-source LAPE to promote more research in this space.

## How to use?

### Setup
```
conda create -n lape -y python=3.8
conda activate lape
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Upstream SSL Pre-training

Its simple! First create a csv file with a single column named *files*. That column should have absolute paths to your raw wav audio files. Next, for upstream SSL pre-training using un-labeled data just run:

```
python train_upstream.py --input /path/to/csv/ --upstream name_of_upstream
```

The `name_of_upstream` should be the name of the upstream algorithm you want to use. The upstreams we currently support are: [DeLoRes](https://arxiv.org/abs/2203.13628), [SLICER](https://ieeexplore.ieee.org/document/10096970), [MAST](https://arxiv.org/abs/2211.01515) and [UNFUSED](https://arxiv.org/abs/2303.05668) Additionally, all other defaults (encoder, learning rate, etc.) are set in the upstream specific configs. An example can be seen in `slicer/config.yaml`. Feel free to change it according to your needs! 

**Note:** For [MAST](https://arxiv.org/abs/2211.01515) you need to change the `base_encoder:` `type` in config to `MAST`.

### Downstream Task Fine-tuning

Once you have pre-trained your encoder using SSL, fine-tune it for any task using this command:

```
python train_downstream.py --input name_of_task --ckpt /path/to/pretrained/ckpt
```

If the task is supported by `HuggingFace🤗` datasets library, the script automatically downloads the data. If not, you need to specify additional arguments including `--train_csv`, `--valid_csv` and `--test_csv`. The csvs should have 2 columns including *wav_path* and *label* where the former is the path to the raw wav file and the latter is the tag of the audio. Similar to upstream, we maintain a config file for all other defaults. An example can be seen in `src/downstream/downstream_config.yaml`.

Also feel free to remove the `--ckpt` argument if you want to train an encoder on your downstream encoder from *scratch* (no SSL pre-training).

## Contribution Guidelines
If you want to contribute (models or algorithms or a completely new feature!) please feel free to open an issue followed by a pull-request.

## Found a bug?
Please raise an issue and we will try our best to resolve it as-soon-as-possible!

## Cite
If you find this toolkit useful, please consider citing following papers.

If you have used the **LAPE** toolkit in your experiments, or the **DeLoRes-S** or **DeLoRes-M** SSL pre-training algorithms:
```
@ARTICLE{9868132,
  author={Ghosh, Sreyan and Seth, Ashish and Umesh, S},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Decorrelating Feature Spaces for Learning General-Purpose Audio Representations}, 
  year={2022},
  volume={16},
  number={6},
  pages={1402-1414},
  doi={10.1109/JSTSP.2022.3202093}}
```

If you have used the **MAST** audio encoder for your work:
```
@inproceedings{ghosh2023mast,
  title={MAST: Multiscale Audio Spectrogram Transformers},
  author={Ghosh, Sreyan and Seth, Ashish and Umesh, S and Manocha, Dinesh},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  organization={IEEE}
}
```

For **SLICER**, **UNFUSED** and **DECAR**:
```
@inproceedings{seth2023slicer,
  title={SLICER: Learning universal audio representations using low-resource self-supervised pre-training},
  author={Seth, Ashish and Ghosh, Sreyan and Umesh, S and Manocha, Dinesh},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  organization={IEEE}
}
```
```
@inproceedings{seth2023unfused,
  title={UNFUSED: UNsupervised Finetuning Using SElf supervised Distillation},
  author={Seth, Ashish and Ghosh, Sreyan and Umesh, S and Manocha, Dinesh},
  booktitle={ICASSP 2023-2023 SASB Workshop},
  year={2023},
  organization={IEEE}
}
```
```
@article{ghosh2021deep,
  title={Deep clustering for general-purpose audio representations},
  author={Ghosh, Sreyan and Katta, Sandesh V and Seth, Ashish and Umesh, Srinivasan},
  journal={arXiv preprint arXiv:2110.08895},
  year={2021}
}
```

## Contact and Contributors
Ashish Seth (email: cs20s030@smail.iitm.ac.in)
[Sreyan Ghosh](https://sreyan88.github.io/) (email: sreyang@umd.edu)
