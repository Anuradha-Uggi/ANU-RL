
# ANU-RL: A New Perspective on Weakly-Supervised Representation Learning for Visual Place Recognition
Code for the TMLR 2026 paper [ANU-RL: A New Perspective on Weakly-Supervised Representation Learning for Visual Place Recognition](https://openreview.net/pdf?id=mXE4OP55il). Most code snippets are borrowed from [MixVPR](https://github.com/amaralibey/mixvpr), [BoQ](https://github.com/amaralibey/Bag-of-Queries), [MSim](https://github.com/msight-tech/research-ms-loss). 

<img src="anu-rl-fig.png" width="100%" height="100%"> 

### Prepare the data and the pretrained model 

The following script will prepare the [CUB](http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) dataset for training by downloading to the ./resource/datasets/ folder; which will then build the data list (train.txt test.txt):

```bash
./scripts/prepare_cub.sh
```

Download the imagenet pretrained model of 
[bninception](http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth) and put it in the folder:  ~/.torch/models/.


### Installation

```bash
pip install -r requirements.txt
python setup.py develop build
```
###  Train and Test on CUB200-2011 with MS-Loss

```bash
./scripts/run_cub.sh
```
Trained models will be saved in the ./output/ folder if using the default config.

Best recall@1 higher than 66 (65.7 in the paper).

### Contact

For any questions, please feel free to reach 
```
github@malongtech.com
```

### Citation

If you use this method or this code in your research, please cite as:

    @inproceedings{wang2019multi,
    title={Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning},
    author={Wang, Xun and Han, Xintong and Huang, Weilin and Dong, Dengke and Scott, Matthew R},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={5022--5030},
    year={2019}
    }

## License

MS-Loss is CC-BY-NC 4.0 licensed, as found in the [LICENSE](LICENSE) file. It is released for academic research / non-commercial use only. If you wish to use for commercial purposes, please contact sales@malongtech.com.

