# Self-training solutions for the ICCV 2023 GeoNet Challenge

## Team Name: CASIA-TIM

Members: Lijun Sheng, Zhengbo Wang, Jian Liang (Contact author)

Affiliation: Institute of Automation, Chinese Academy of Sciences (CASIA)

### File structure:
```
|–– readme.md
|–– data_list/
|   |–– UNIDA/
|	|	|–– usa_train.txt
|	|	|–– asia_train.txt
|	|	|–– asia_test.txt
|	|	|–– test.txt
|   |–– OBJ/
|   |–– PLACE/
|   
|–– main_unida.py
|–– main_places.py
|–– main_imnet.py
|–– data_list.py
|–– network.py
```

### Prerequisites:
- python == 3.10.6
- torch ==1.12.0
- torchvision == 0.13.0
- numpy, scipy, sklearn, PIL, argparse

### Dataset:
We use the dataset provided by the challenge to generate txt files and place them in the data_list folder according to the names of each dataset (i.e., UNIDA, OBJ, PLACE). If you want to run the code, please **modify the absolute paths** in all files under data_list folder.

### Note:
We integrate the source model training, model adaptation, and test file generation in single python code. The test file of the source model is saved as source_test.txt, and the test file based on the adaptive model is saved as **target_test.txt**.

### Training:

1. ##### GeoUniDA
	```python
		python main_unida.py --dset UNIDA --gpu_id 0 
	```

2. ##### GeoImNet
	```python
		python main_imnet.py --dset OBJ --gpu_id 1 
	```

3. ##### GeoPlace
	```python
		python main_place.py --dset PLACE --gpu_id 2 
	```


### Contact

- [liangjian92@gmail.com](mailto:liangjian92@gmail.com)

