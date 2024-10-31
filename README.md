# LDCformer_Ext

### Enhancing Learnable Descriptive Convolutional Vision Transformer for Face Anti-Spoofing  

This is the extension version from our previous paper in  LDCformer: Incorporating Learnable Descriptive Convolution to Vision Transformer for Face Anti-Spoofing (ICIP '23) 

## Illustration of the architecture of our extended version of LDCformer.
![plot](figures/framework.png)

## Requirements
```
grad_cam==1.3.5
matplotlib==3.5.2
numpy==1.22.3
scikit_learn==1.1.2
torch==1.12.0
torchvision==0.13.0
```

## Training
Step 1: run `Amap_train.py` to get pretrained model for producing activation map 

Step 2: run `train.py` to train LDCNet

## Testing
run `test.py`

## Citation

If you use the LDCformer, please cite the paper:

 ```
@inproceedings{huang2023ldcformer,
  title={LDCformer: Incorporating Learnable Descriptive Convolution to Vision Transformer for Face Anti-Spoofing},
  author={Huang, Pei-Kai and Chiang, Cheng-Hsuan and Chong, Jun-Xiong and Chen, Tzu-Hsien and Ni, Hui-Yu and Hsu, Chiou-Ting},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={121--125},
  year={2023},
  organization={IEEE}
}
 @inproceedings{huang2022learnable,
  title={Learnable Descriptive Convolutional Network for Face Anti-Spoofing},
  author={Huang, Pei-Kai and H.Y. Ni and Y.Q. Ni and C.T. Hsu},
  booktitle={BMVC},
  year={2022}
}
```

## Contact us
We are students from MPLAB at National Tsing Hua University.  
Huang, Pei-Kai <alwayswithme@gapp.nthu.edu.tw>  
