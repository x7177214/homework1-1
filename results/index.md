# CEDL2017 HW1 Report <span style="color:red">(id)</span>
Author: Hank Lu (呂賢鑫) 105061585

#Project 5: Deep Classification

## Overview
The project is related to 
> quote


## Implementation
1. One
	* item
	* item
2. Two

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


