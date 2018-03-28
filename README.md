# Project: Dogs, Fried Chicken or Blueberry Muffins?
![image](figs/chicken.jpg)
![image](figs/muffin.jpg)

### [Full Project Description](doc/project3_desc.md)

Term: Spring 2018

+ Team #4
+ Team members
	+ Lin, Yanjun
	+ Jiang, Chenfei
	+ Tao, Wenyi
	+ Wan, Qianhui
	+ Yao, Jingtian

+ Project summary: In this project, we created a classification engine for 3000 images of dogs versus fried chicken versus blueberry muffins using various model evaluation and selection methods. The primary concerns and evaluation criterion are around computational efficiency and memory cost. We found that MobileNet has a very high validation accuracy and it is relatively computational efficient when comparing to other CNN structures. This model outperforms all other non-deep-learning models in accuracy, and it is not computational costy, which means it could be applied to the mobile device very efficiently.
For feature extraction, our team compared various methods including ORB, SIFT,SURF, RGB about the performance regarding the feature dimensions and extraction time. On top of the base model using SIFT feature extration and Gradient Boosting method, we applied a series number of advanced models including LinearSVM, RBF kernal SVM, XGBoost.

	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
