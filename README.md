## AlphaPose
Alpha Pose is a accurate multi-person pose estimation system. It is the first open-sourced system that can achieve 70+ mAP (73.2 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset

## Contents
1. [AlphaPose](#alphapose)
2. [Results](#results)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Output](#output)
6. [Speeding Up Alpha Pose](#speeding-up-alpha-pose)
7. [Contributors](#contributors)
9. [Citation](#citation)
10. [License](#license)



## Results
Results on COCO test-dev 2015:
<center>

| Method | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|
| OpenPose (CMU-Pose) | 61.8 | 84.9 | 67.5 | 57.1 | 68.2 |
| Detectron (Mask R-CNN) | 67.0 | 88.0 | 73.1 | 62.2 | 75.6 |
| **AlphaPose** | **61.8** | **83.7** | **69.8** | **69.8** | **69.8** |

</center>

Results on MPII full test set:
<center>

| Method | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Ave |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| OpenPose (CMU-Pose) | 91.2 | 87.6 | 77.7 | 66.8 | 75.4 | 68.9 | 61.7 | 75.6 |
| Newell & Deng | **92.1** | 89.3 | 78.9 | 69.8 | 76.2 | 71.6 | 64.7 | 77.5 |
| **AlphaPose** | 91.3 | **90.5** | **84.0** | **76.4** | **80.3** | **79.9** | **72.4** | **82.1** |

</center>

### Pose Estimation
<p align="center">
    <img src="doc/pose.gif", width="360">
</p>

### Pose Tracking


## Installation
1. Get the code and build related modules.
  ```Shell
  git clone https://github.com/MVIG-SJTU/AlphaPose.git
  cd AlphaPose/human-detection/lib/
  make clean
  make
  cd newnms/
  make
  cd ../../../
  ```
2. Install [Torch](https://github.com/torch/distro) and [TensorFlow](https://www.tensorflow.org/install/)(verson >= 1.2).
  ```Shell
  chmod +x install.sh
  ./install.sh
  ```
3. Run fetch_models.sh to download our pre-trained models.
  ```Shell
  chmod +x fetch_models.sh
  ./fetch_models.sh
  ```

## Quick Start
- **Demo**:  Run AlphaPose for all images in a folder and visualize the results with:
```
./run.sh --indir examples/demo/ --outdir examples/results/ --vis
```

The visualized results will be stored in examples/results/RENDER. To easily process images/video and display/save the results, please see [doc/run.md](doc/run.md).

## Output
Output (format, keypoint index ordering, etc.) in [doc/output.md](doc/output.md).

## Speeding Up AlphaPose
We provide a `fast` mode for human-detection that disables multi-scale tesing. You can turn it on by adding `--mode fast`.

And if you have multiple gpus on your machine or have large gpu memories, you can speed up the pose estimation step by using multi-gpu testing or large batch tesing with:
```
./run.sh --indir examples/demo/ --outdir examples/results/ --gpu 0,1,2,3 --batch 5
```
It assumes that you have 4 gpu cards on your machine and *each card* can run a batch of 5 images. See [doc/run.md](doc/run.md) for more details.

## Contributors
The main contributors are listed in [doc/contributors.md](doc/contributors.md).

## Citation
Please cite these papers in your publications if it helps your research:

    @inproceedings{fang2017rmpe,
      title={{RMPE}: Regional Multi-person Pose Estimation},
      author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
      booktitle={ICCV},
      year={2017}
    }



## License
AlphaPose is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, contact [Cewu Lu](http://mvig.sjtu.edu.cn/)