# ConClu
Unsupervised Point Cloud Pre-training via Contrasting and Clustering

### Abstract
The annotation for large-scale point clouds is still time-consuming and unavailable for many complex real-world tasks. Point cloud pre-training is a promising direction to auto-extract features without labeled data. Therefore, this paper proposes a general unsupervised approach, named ConClu, 
for point cloud pre-training by jointly performing contrasting and clustering. Specifically, the contrasting is formulated by maximizing the similarity feature vectors produced by encoders fed with two augmentations of the same point cloud. The clustering simultaneously clusters the data while enforcing consistency between cluster assignments produced different augmentations. Experimental evaluations on downstream applications outperform state-of-the-art techniques, which demonstrates the effectiveness of our framework.

### Citation

If you take use of our code or feel our paper is useful for you, please cite our papers:

```
@inproceedings{mei2022unsupervised,
  title={Unsupervised Point Cloud Pre-Training Via Contrasting and Clustering},
  author={Mei, Guofeng and Huang, Xiaoshui and Liu, Juan and Zhang, Jian and Wu, Qiang},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={66--70},
  year={2022},
  organization={IEEE}
}
```

```
@article{mei2022unsupervised,
  title={Unsupervised learning on 3d point clouds by clustering and contrasting},
  author={Mei, Guofeng and Yu, Litao and Wu, Qiang and Zhang, Jian and Bennamoun, Mohammed},
  journal={arXiv preprint arXiv:2202.02543},
  year={2022}
}
```

If you have any questions, please contact me without hesitation (gfmeiwhu@outlook.com).
