# Composition and Content-Based Image Retrieval(CCBIR)

This is the official webpage of the paper "Enhancing Historical Image Retrieval with Compositional Cues", accepted to AIRoV – The First Austrian Symposium on AI, Robotics, and Vision

Arxiv: TBD

**News**:

- **Mar. 20, 2024**: We release the code for CCBIR.

## 1 Datasets

Our work is divided into two main segments: the task of image composition classification and the task of image retrieval. 

The dataset employed for the image composition classification task is the [KU-PCP dataset](http://mcl.korea.ac.kr/research/Submitted/jtlee_JVCIR2018/), introduced by J.-T. Lee et al.

For the image retrieval task, we extract data from the [HISTORIAN dataset](https://zenodo.org/records/6644516), which is richly annotated. The extraction process is shown as follows, and for more details, please refer to our paper:

<img src="./img/Picture1.png"/>


## 2 Results

Visualisation of KCM effect. The top row features the original grayscale images, and the bottom row highlights the KCM, pinpointing key compositional areas as detected by our model:

<img src="./img/Picture3.png"/>

Scatter plot and histogram of positive and negative samples when $L_{KCM}$ is 0.5:

<img src="./img/Picture4.png"/>

Comparison of retrieval results with different $L_{KCM}$. We selected only the central frame image from each shot in the test set as the target database for retrieval, returning the five highest similarity-scored images for a single image query:

<img src="./img/Picture5.png"/>


## 3 Implementation

The implementation of the CAM and KCM mechanisms within CCNet draws significantly from [CACNet-Pytorch](https://github.com/bo-zhang-cs/CACNet-Pytorch).

### 3.1 File Structure

```
CCBIR:.
|   
+---CBIRNet
|       dataset.py
|       network.py
|       network_cc.py
|       network_module.py
|       select_frames.py
|       test.py
|       test_one.py
|       test_vis.py
|       train.py
|       trainer.py
|       transform.py
|       utils.py
|       validation.py
|       __init__.py
|       
+---CCNet
|       dataset.py
|       network.py
|       network_module.py
|       train.py
|       trainer.py
|       transform.py
|       utils.py
|       validation.py
|       __init__.py
```

### 3.2 Required Libraries

Please refer to requirements.txt

## 4 Network Architectures

<img src="./img/Picture2.png"/>


## 5 Citation

If you find this work useful for your research, please cite:

```bash
TBD
```

Please contact tylin@cvl.tuwien.ac.at for further questions.
