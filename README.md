# AMESAC #
This is the official implementation of the paper: "Multi-Task Reinforcement Learning With Attention-Based Mixture of Experts." IEEE Robotics and Automation Letters (2023).

### Installation ###
Firstly, create a conda environment as following:
```
conda create -n AMESAC python=3.8
```
Then, install the Metaworld simulator as following:
```
git clone https://github.com/Farama-Foundation/Metaworld.git
cd Metaworld
pip install -e .
```
And download our project and install the dependencies:
```
git clone https://github.com/123penny123/AMESAC.git
cd AMESAC
pip install -r requirements.txt
```
### Usage ###
To train and evaluate the AMESAC performance on the MT-10 or MT_50 asks, run the following code:
```
python train_v2_mt10.py 
python train_v2_mt50.py
```
our hyperparameters are saved in the **./cfg/** directory.
You can easily use the tensorboard to monitor the training process.

### Citation ###
If you use AMESAC in your project, please cite our paper:
```
@article{Cheng2023MultiTaskRL,
  title={Multi-Task Reinforcement Learning With Attention-Based Mixture of Experts},
  author={Guangran Cheng and Lu Dong and Wenzhe Cai and Changyin Sun},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  volume={8},
  pages={3811-3818},
  url={https://api.semanticscholar.org/CorpusID:258641316}
}

```