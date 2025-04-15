# MapFlow

**Official implementation of the paper:\
_MapFlow: Multi-Agent Pedestrian Trajectory Prediction Using Normalizing Flow_**

## 📁 Repository Structure

This repository includes the following directories:

- `config/` – Configuration files for training runs.
- `dataset/` – Contains the raw ETH/UCY datasets.
- `src/` – Source code for the model and utilities.
- `weights/` – Pretrained weights of the autoencoder.

## ⚙️ Setup

To install all necessary dependencies, run the following from the project root:

```bash
pip install -r requirements.txt
```


## 🏋️‍♂️ Training

Training is scenario-specific using the leave-one-out approach on the ETH/UCY dataset. Each scenario has its own config file. More settings can be found in config/settings.py.

For example, to train on the eth_uni scene, run:

```bash
python main.py --flagfile config/config_eth_uni.cfg
```


- Model weights will be saved in: `weights/normalizingflow/`
- Results will be stored in: `results/`

## 📄 Citation

If you find this work useful in your research, please consider citing:\
@inproceedings{stefani2024mapflow,\
&nbsp;&nbsp;&nbsp;&nbsp;  title={MapFlow: Multi-Agent Pedestrian Trajectory Prediction Using Normalizing Flow},\
&nbsp;&nbsp;&nbsp;&nbsp;  author={Stefani, Antonio Luigi and Bisagno, Niccol{\o} and Conci, Nicola},\
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},\
&nbsp;&nbsp;&nbsp;&nbsp;  pages={3295--3299},\
&nbsp;&nbsp;&nbsp;&nbsp;  year={2024},\
&nbsp;&nbsp;&nbsp;&nbsp;  organization={IEEE}\
}

## 🛠 License

This project is licensed under the Creative Commons Zero v1.0 Universal license (CC0 1.0).
You are free to use, modify, and distribute this software without restriction.

See the LICENSE file for full details, or visit [creativecommons.org/publicdomain/zero/1.0](https://creativecommons.org/publicdomain/zero/1.0).
