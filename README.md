# ACVAE-VC

This repository provides an official PyTorch implementation for [ACVAE-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/acvae-vc3/index.html).

ACVAE-VC is a non-parallel many-to-many voice conversion (VC) method using an auxiliary classifier variational autoencoder (ACVAE). The current version performs VC by first modifying the mel-spectrogram of input speech, and then generating a waveform using a speaker-independent neural vocoder (HifiGAN or Parallel WaveGAN) from the modified spectrogram.

Audio samples are available [here](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/acvae-vc3/index.html).

## Paper

[Hirokazu Kameoka](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/index-e.html), [Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/index.html), [Kou Tanaka](http://www.kecl.ntt.co.jp/people/tanaka.ko/index.html), Nobukatsu Hojo, "**ACVAE-VC: Non-Parallel Voice Conversion With Auxiliary Classifier Variational Autoencoder**," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 27, no. 9, pp. 1432-1443, Sep. 2019. [**[Paper]**](https://ieeexplore.ieee.org/abstract/document/8718381) 



## Preparation

#### Prerequisites

- See `requirements.txt`.

#### Dataset

1. Setup your training and test sets. The data structure should look like:

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
       
   /path/to/dataset/test
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

#### Waveform generator

1. Place a copy of the directory `parallel_wavegan` from https://github.com/kan-bayashi/ParallelWaveGAN in `hifigan/` (or `pwg/`).
2. HifiGAN models trained on several databases can be found [here](https://drive.google.com/drive/folders/1RvagKsKaCih0qhRP6XkSF07r3uNFhB5T?usp=sharing). Once these are downloaded, place them in `hifigan/egs/`. Please contact me if you have any problems downloading.
3. Optionally, Parallel WaveGAN can be used instead for waveform generation. The trained models are available [here](https://drive.google.com/drive/folders/1zRYZ9dx16dONn1SEuO4wXjjgJHaYSKwb?usp=sharing). Once these are downloaded, place them in `pwg/egs/`. 

## Main

#### Train

To run all stages for model training, execute:

```bash
./recipes/run_train.sh [-g gpu] [-a arch_type] [-s stage] [-e exp_name]
```

- Options:

  ```bash
  -g: GPU device (default: -1)
  #    -1 indicates CPU
  -a: VAE architecture type ("conv" or "rnn")
  #    conv: 1D fully convolutional network (default)
  #    rnn: Bidirectional long short-term memory network
  -s: Stage to start (0 or 1)
  #    Stages 0 and 1 correspond to feature extraction and model training, respectively.
  -e: Experiment name (default: "conv_exp1")
  #    This name will be used at test time to specify the trained model.
  ```

- Examples:

  ```bash
  # To run the training from scratch with the default settings:
  ./recipes/run_train.sh
  
  # To skip the feature extraction stage:
  ./recipes/run_train.sh -s 1
  
  # To set the gpu device to, say, 0:
  ./recipes/run_train.sh -g 0
  
  # To use a VAE with a recurrent architecture:
  ./recipes/run_train.sh -a rnn -e rnn_exp1
  ```


To monitor the training process, use tensorboard:

```bash
tensorboard [--logdir log_path]
```

#### Test

To perform conversion, execute:

```bash
./recipes/run_test.sh [-g gpu] [-e exp_name] [-c checkpoint] [-v vocoder]
```

- Options:

  ```bash
  -g: GPU device (default: -1)
  #    -1 indicates CPU
  -e: Experiment name (e.g., "conv_exp1")
  -c: Model checkpoint to load (default: 0)
  #    0 indicates the newest model
  -v: Vocoder type ("hifigan" or "pwg")
  #    hifigan: HifiGAN (default)
  #    pwg: Parallel WaveGAN
  ```

- Examples:

  ```bash
  # To perform conversion with the default settings:
  ./recipes/run_test.sh -g 0 -e conv_exp1
  
  # To use Parallel WaveGAN as an alternative for waveform generation:
  ./recipes/run_test.sh -g 0 -e conv_exp1 -v pwg
  ```

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
