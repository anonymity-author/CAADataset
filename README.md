# A New Multimodal Dataset and Method for Classroom Atmosphere Assessment in Real-world Environment

## About the dataset：
![image](https://github.com/anonymity-author/CAADataset/blob/main/CAAM/fig/dataset.png)
Figure 1. An Overview and distribution of the CAA dataset. More precisely, the Lesson ID denotes the unique number assigned to each classroom video. Each recorded lesson is associated with four feature labels and three modal data. The four labels include classroom atmosphere score, classroom atmosphere binarization (1: pass, 0: failed), regression grade and teacher gender. The three modal data consist of audio, visual, and text, as detailed in Section 3.3. In addition, the distribution of five grades can be observed at the bottom right of the Figure, and they are Excellent, Good, Medium, Poor and Failed.

  The CAA dataset consists of 1500 recorded classroom videos from 1500 teachers in a total of 6 National Teacher Lecture Competitions. As each competition's video covers different levels of teaching scenarios, there are sufficient positive and negative samples. For each classroom video, it contains 9 features, including 6 visual features, 2 audio features, and 1 transcribed text feature. The dataset provides a total of 13500 learnable feature data that are publicly available for classroom atmosphere assessment. Additionally, the dataset covers 9 secondary school subjects, namely Chinese, Mathematics, English, Politics, History, Geography, Physics, Chemistry, and Biology.
  			
Table 1. Details of the training and test sets.
| Dataset Split | Number of samples | Number of learnable features | Duration [h:min:s:] |
| :---: | :---: | :---: | :---: | 
|Training set|	1200 |	10800	|223:30:30|
|Test set |	300| 2700 | 56:4:32 |
| All | 1500 |13500|	278:35:2|

1500 teaching videos which represent by low-level features are partitioned into 1200 training samples and 300 test samples, following an approximate split ratio of 8:2. The training set consists of 222 hours, 30 minutes, and 30 seconds of video, while the test set comprises a total duration of 56 hours, 4 minutes, and 32 seconds. Detailed information is shown in Table 1.

## About the CAAM：

### Requirement
Python >= 3.6

Pytorch >=1.8.0

### Dataset Preparation
**CAA dataset**

If the article is accepted for publication, you can download our prepared CAA dataset from ["Google Drive"]. Then, please move the uncompressed data folder to `./CAAM/data`. 

Then, run the following commands to preprocess the dataset:

` python ./CAAM/caa_preprocessing/database_generation_train.py ` 

` python ./CAAM/caa_preprocessing/database_generation_test.py ` 

### Training & Evaluation

This paper proposes a simple yet effective Classroom Atmosphere Assessment Method (CAAM) based on the CAA dataset. By modeling and fusing emotional features from audio, visual, and textual modalities, CAAM maps teacher emotions to classroom atmosphere scores.

To train and evaluate the model on the CAA dataset, simply run: 

` python main.py `


If you use the CAA dataset, please cite this paper: A New Multimodal Dataset and Method for Classroom Atmosphere Assessment in Real-world Environment.
