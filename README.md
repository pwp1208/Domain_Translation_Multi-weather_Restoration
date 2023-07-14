# Multi-weather Image Restoration via Domain Translation (ICCV2022)
Prashant W Patil, Sunil Gupta, Santu Rana, Svetha Venkatesh, and Subrahmanyam Murala


[![paper](https://img.shields.io/badge/Paper-<COLOR>.svg)](https://drive.google.com/file/d/1itwA0a1JQvS6sVsGDJ8Pt2DnQtG3UrJ8/view?usp=sharing)


<hr />

> **Abstract:** *Weather degraded conditions such as rain, haze, snow, etc. may degrade the performance of most computer vision systems. Therefore, effective restoration of multiweather degraded images is an essential prerequisite for successful functioning of such systems. The current multiweather image restoration approaches utilize a model that is trained on a combined dataset consisting of individual images for rainy, snowy, and hazy weather degradations. These methods may face challenges when dealing with real-world situations where the images may have multiple, more intricate weather conditions. To address this issue, we propose a domain translation-based unified method for multi-weather image restoration. In this approach, the proposed network learns multiple weather degradations simultaneously, making it immune for realworld conditions. Specifically, we first propose an instancelevel domain (weather) translation with multi-attentive feature learning approach to get different weather-degraded variants of the same scenario. Next, the original and translated images are used as input to the proposed novel multi-weather restoration network which utilizes a progressive multi-domain deformable alignment (PMDA) with cascaded multi-head attention (CMA). The proposed PMDA facilitates the restoration network to learn weather-invariant clues effectively. Further, PMDA and respective decoder features are merged via proposed CMA module for restoration. Extensive experimental results on synthetic and realworld hazy, rainy, and snowy image databases clearly demonstrate that our model outperforms the state-of-the-art multi-weather image restoration methods.* 
<hr />

## Network Architecture

<img src = 'Overview.jpg'> 

## Requirements:

	Anaconda

	Pytorch > 1.8


## Testing Images
	Keep Testing Images in "dataset" folder.

## Checkpoints:
	Keep the checkpoints in "./checkpoints/"


## Databases
	1. Outdoor_rain
	2. CSD
	3. SOTS-outdoor
	4. Real-world


## For testing on any database:
	1. Run the "test.py" file as python test.py
	2. The results will be stored in "results/" folder


## Citation
If our method is useful for your research, please consider citing:
    
    @inproceedings{DTRestoration2023,
	title={Multi-weather Image Restoration via Domain Translation},
	author={Prashant W Patil, Sunil Gupta, Santu Rana, Svetha Venkatesh, and Subrahmanyam Murala},
	booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
	pages={Accepted on 14/07/2023},
	year={2023}

}


## Contact
Please contact prashant.patil@deakin.edu.au, if you are facing any issue.


