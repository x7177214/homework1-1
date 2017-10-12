# CEDL2017 HW1 Report: Deep Classification <span style="color:red">(id)</span>
Author: Hank Lu (呂賢鑫) 105061585

## Overview
The project is related to classification using Tensorflow and modified from the code of [wenxinxu/ResNeXt-in-tensorflow] (https://github.com/wenxinxu/ResNeXt-in-tensorflow)
- `main.py`: Do training or testing.
- `data_input.py`: Processing data io.
- `hyper_parameters.py`: Set up the hyper parameters.
- `resNeXt.py`: Define the [ResNext](https://arxiv.org/pdf/1611.05431.pdf) model.

## Implementation
### Model Architectures
This project use the [ResNext](https://arxiv.org/pdf/1611.05431.pdf) to do classification. 
![](https://github.com/x7177214/homework1-1/blob/oh%2Cmfc%2Cobj%2Bges/results/arch.png)

```
Code highlights
```

## Installation
* [pandas](http://pandas.pydata.org/)
* [scikit-image](http://scikit-image.org/docs/dev/install.html)

1. Download this project code.
2. Download [dataset](https://drive.google.com/drive/folders/0BwCy2boZhfdBdXdFWnEtNWJYRzQ)(`frames/` and `labels/`) and place them to this project folder `dataset/`.
3. Convert the training data to tfRecord format to speed up the training (cost 46 GB space). `python data_input`
4. For training from script: `python main.py --mode=train --version='model_1'` The training logs, checkpoints, and error.csv file will be saved in the folder with name logs_$version
4. For testing: 
* replace the test_ckpt_path FLAGS in `hyper_parameters.py` with your path to the ckpt (ex: 'logs_onlyhand_c=3_b=15/model.ckpt-39999')
* then run `python main.py --mode=test`


### Results

<table border=1>
<tr>
<td>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

<tr>
<td>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg"  width="24%"/>
<img src="placeholder.jpg" width="24%"/>
<img src="placeholder.jpg" width="24%"/>
</td>
</tr>

</table>


