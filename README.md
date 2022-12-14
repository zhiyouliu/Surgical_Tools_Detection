# Surgical Tool Object Detection

Cooperated with Pengyu Kan, Hanting Fei and Lelin Zhong.

The dataset is available at [here](https://www.kaggle.com/datasets/dilavado/labeled-surgical-tools)

The Final report is available at [here](https://livejohnshopkins-my.sharepoint.com/:b:/g/personal/zliu167_jh_edu/ETJS1ko432hDnWfkwl_8aqsBi7vwgCzb0cch-uuclnOP8w?e=F0Ubka)

The slide for presentation is available [here](https://livejohnshopkins-my.sharepoint.com/:b:/g/personal/zliu167_jh_edu/EUlMdnjTv1FOnjlUyQwOkF0BMsGf0nUyreuYIJoP0ywRLg?e=6bGWB8)

This project focuses developping effective object detector to detect surgical tools. We specifically study on the capability of transformer based architecture for surgical tool dataset.

We provide an example of implementation for the dataset class from pytorch in ```surgical_tool_dataset.ipynb```.

## Instructions for running Deformable DETR:

1. You should download the dataset and save it in ```./Deformable-DETR-surgical_tool/data``` for properly run.

2. To run the Deformable DETR, remember to build up the required dependencies with following processes:
	- We recommend using conda with ```python==3.8```. Then, you should firstly run:
	
		```
		conda install cudatoolkit=10.0
		```

		since the Deformable DETR has operations directly over CUDA and only supports for $\approx$ cuda 10.0 for properly import of its module. Though you can run with other versions of torch in your system and environment. Thus, we recommend installing this toolkit into the environment you are working on.

		Also, Deformable DETR requires ```gcc 5+``` for properly importing its module. If the gcc version is too low, please use:
		
		```
		conda install -c psi4 gcc-5
		``` 
		
		or 

		```
		conda install https://anaconda.org/brown-data-science/gcc/5.4.0/download/linux-64/gcc-5.4.0-0.tar.bz2
		```

	- Then, you should install torch and torchvision:

		```
		pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
		```

		choose the correct ```+cu101``` version depending on your cuda version. Acutally, since we are using the ```cudatoolkit=10.0```, it currently runs properly with torch version ```+cu101```. 

	- Next, follow the ```./Deformable-DETR-surgical_tool/README.md```. to properly run the ```bash ./Deformable-DETR-surgical_tool/models/ops/make.sh``` file, to build up the deformable DETR into the system.

		Test with ```python ./Deformable-DETR-surgical_tool/models/ops/test.py``` to make sure you have properly built up.

	- Now, build up the rest dependent packages:

		```
		pip install -r ./Deformable-DETR-surgical_tool/requirements_surgical_tool.txt --no-dependencies
		```

3. Install the pretrained deformable DETR checkpoint from:
   - https://drive.google.com/file/d/1nDWZWHuRwtwGden77NLM9JoWe-YisJnA/view  for base deformable DETR

   - https://drive.google.com/file/d/1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI/view for deformable DETR with box refinement architecture

   Save these checkpoint into  ```./Deformable-DETR-surgical_tool/pretrain_ckpt```

4. Now, you should be able to properly run:
	
	```bash ./Deformable-DETR-surgical_tool/train_surgical_tool.sh```

	You can remove the tag of ```--debug``` for a fully running.

5. For running the box refinement architecture:

	```bash ./Deformable-DETR-surgical_tool/train_with_box_refine.sh```


