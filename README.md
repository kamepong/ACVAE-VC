# ACVAE-VC

This repository provides an official PyTorch implementation for [ACVAE-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/acvae-vc2/index.html).

ACVAE-VC is a non-parallel many-to-many voice conversion method that modifies the mel-spectrogram of input speech, and generates a waveform from the modified spectrogram using Parallel WaveGAN.

Audio samples are available [here](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/acvae-vc2/index.html).

#### Prerequisites

- See `requirements.txt`.

- See https://github.com/kan-bayashi/ParallelWaveGAN for the packages needed to set up Parallel WaveGAN.

  

## Paper

[Hirokazu Kameoka](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/index-e.html), [Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/index.html), [Kou Tanaka](http://www.kecl.ntt.co.jp/people/tanaka.ko/index.html), Nobukatsu Hojo, "**ACVAE-VC: Non-Parallel Voice Conversion With Auxiliary Classifier Variational Autoencoder**," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 27, no. 9, pp. 1432-1443, Sep. 2019. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/8718381) 



## Preparation

#### Dataset

1. Setup your training dataset. The data structure should look like:

   ```bash
   /path/to/dataset/training
   ├── spk_1
   │   ├── utt1.wav
   │   ...
   ├── spk_2
   │   ├── utt1.wav
   │   ...
   └── spk_N
       ├── utt1.wav
       ...
   ```

#### Parallel WaveGAN

1. Setup Parallel WaveGAN.  Place a copy of the directory `parallel_wavegan` from https://github.com/kan-bayashi/ParallelWaveGAN in `pwg/`.
2. Parallel WaveGAN models trained on several databases can be found [here](https://app.box.com/folder/127558077224). Once these are downloaded, place them in `pwg/egs/`. Please contact me if you have any problems downloading.

```bash
# Model trained on the ATR database (11 speakers)
cp -r ATR_all_flen64ms_fshift8ms pwg/egs/
# Model trained on the CMU ARCTIC dataset (4 speakers)
cp -r arctic_4spk_flen64ms_fshift8ms pwg/egs/
```



## Main

See shell scripts in `recipes` for examples of training on different datasets.

#### Feature Extraction

To extract the normalized mel-spectrograms from the training utterances, execute:

```bash
python extract_features.py
python compute_statistics.py
python normalize_features.py
```

#### Train

To train the model, execute:

```bash
python main.py [-g gpu] [-arc arch_type] [-exp exp_name] ...
```

- Options:
  - -g: GPU device# (CPU will be used if not specified)
  - -arc: Architecture type ("conv" for fully convolutional and "rnn" for recurrent architectures)
  - -exp: Experiment name (e.g., "conv_exp1")


To monitor the training process, use tensorboard:

```bash
tensorboard [--logdir log_path]
```



#### Test

To perform conversion, execute:

```bash
python convert.py [-g gpu] [-exp exp_name] [-ckpt checkpoint] ...
```

- Options:
  - -g: GPU device# (CPU will be used if not specified)
  - -exp: Experiment name (e.g., "conv_exp1")
  - -ckpt: Model checkpoint (The latest model will be selected if not specified)

  

## Citation

If you find this work useful for your research, please cite our paper.

```
@Article{Kameoka2019IEEETrans_ACVAE-VC,
  author={Hirokazu Kameoka and Takuhiro Kaneko and Kou Tanaka and Nobukatsu Hojo},
  title={{ACVAE-VC}: Non-Parallel Voice Conversion With Auxiliary Classifier Variational Autoencoder},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={27},
  number={9},
  pages={1432--1443},
  year=2019
}
```



## Author

Hirokazu Kameoka ([@kamepong](https://github.com/kamepong))

E-mail: kame.hirokazu@gmail.com
