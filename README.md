## Predictive modeling for autism spectrum disorder
Early Prediction of Austim spectrum disorder among people using machine learning models

## About
Predictive modeling for autism spectrum disorder (ASD) involves the use of statistical techniques and machine learning algorithms to analyze various factors and predict the likelihood of ASD diagnosis or identify patterns related to ASD traits. These models can help in early detection, intervention planning, and understanding the underlying mechanisms of ASD.

## Features

- Stratification of data
- Multi model data integration.
- High scalability.
- Less time complexity.
- cost efficient

## Requirements

* Operating System: Requires a 64-bit OS (Windows 10 or Ubuntu) for compatibility with deep learning frameworks.
* Development Environment: Python 3.6 or later is necessary for coding the sign language detection system.
* Deep Learning Frameworks: TensorFlow for model training, MediaPipe for hand gesture recognition.
* Image Processing Libraries: OpenCV is essential for efficient image processing and real-time hand gesture recognition.
* Version Control: Implementation of Git for collaborative development and effective code management.
* IDE: Use of VSCode as the Integrated Development Environment for coding, debugging, and version control integration.
* Additional Dependencies: Includes scikit-learn, TensorFlow (versions 2.4.1), TensorFlow GPU, OpenCV, and Mediapipe for deep learning tasks.

## System Architecture


![image](https://github.com/SAFZZ/Projectwork2/assets/75234912/069a10c1-22e0-4808-8bb2-351955533faf)

## Code
## Importing the necessary libraries
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns
 ## Reading the data
 train = pd.read_csv('train.csv')
 test = pd.read_csv('test.csv')
 ## Finding the numerical and categorical columns
 cat_cols = [feature for feature in train.columns if
 train[feature].dtypes == 'O']
 num_cols = [feature for feature in train.columns if feature not in
 cat_cols]
 ## Checking the column names in the dataframe
 train.columns
 ## Printing the numerical and categorical columns
 print(f"Numerical columns: {num_cols}\n")
 print(f"Categorical columns: {cat_cols}")
 ## Finding the unique values in each of the categorical feature columns
 for feature in cat_cols:
 print(f"{feature}:")
 print(f"Number of unique values in the {feature}:
 {train[feature].nunique()}")
 print(f"Unique values: {train[feature].unique()}")
 print('\n')
 
## Output


#### Output1 - Traning and validation accuracy
![image](https://github.com/SAFZZ/Projectwork2/assets/75234912/58cfe6da-036c-49b1-9ec2-097653515d6a)


#### Output2 - Heatmap
![image](https://github.com/SAFZZ/Projectwork2/assets/75234912/871fd26f-5d30-45eb-a215-0d462a56c5f9)




## Results and Impact
In conclusion, the autism prediction model utilizing various classifiers such as Gradient Boosting, Support Vector Classifier (SVC), Random Forest, K-Nearest Neighbors (KNN), among others, has demonstrated remarkable efficacy and robustness in its performance. The amalgamation of these classifiers has resulted in a model that not only exhibits high accuracy but also provides insightful metrics such as confusion matrices and heatmaps for better understanding and interpretation.Through meticulous training and validation, the model has showcased its capability to accurately predict autism spectrum disorder (ASD) based on the input features. The utilization of multiple classifiers adds layers of sophistication, enabling the model to capture diverse patterns within the data and make more nuanced predictions.

## Articles published / References
 [1] S. Bhat, U. R. Acharya, H. Adeli, G. M. Bairy, and A. Adeli, “Autism: cause factors, early diagnosis and therapies,” Reviews in the Neurosciences, vol. 25, no. 6, pp. 841–850, 2014.
 [2}J. L. Matson, N. C. Turygin, J. Beighley, R. Rieske, K. Tureck, and M. L. Matson, “Applied behavior analysis in autism spectrum disorders: recent developments, strengths, and pitfalls,”
 Research in Autism Spectrum Disorders, vol. 6, no. 1, pp. 144–150, 2012.
 [3] U. Frith and F. Happé, “Autism spectrum disorder,” Current Biology, vol. 15, no. 19, pp.R786–R790, 2005.
 [4] I. Rapin, “The autistic-spectrum disorders,” New England Journal of Medicine, vol. 347, no. 5, pp.
 302-303, 2002.
 [5] P. J. Landrigan, “What causes autism? Exploring the environmental contribution,” Current Opinion in Pediatrics, vol. 22, no. 2, pp. 219–225, 2010.
 [6] J. S. Sutcliffe, “Insights into the pathogenesis of autism,” Science, vol. 321, no. 5886, pp. 208-209,
 2008.AUTISM TREATMENT CENTER OF AMERICA, What is autism? 2021,https://autismtreatmentcenter.org/knowledge-base/what-is-autism-overview/?gclid=EAIaIQob
 ChMIu7u77Nq46gIVgtKyCh3OAQQhEAAYASAAEgJk9fD_BwE.
 [6] A. Zunino, P. Morerio, and A. Cavallo, “Video gesture analysis for autism spectrum disorder detection,” in Proceedings of the 2018 24th International Conference on Pattern Recognition
 (ICPR), pp. 3421–3426, IEEE, Beijing, China, November 2018.
 [7] G. Dawson and R. Bernier, “A quarter century of progress on the early detection and treatment of
 autism spectrum disorder,” Development and Psychopathology, vol. 25, no. 4pt2, pp.
 1455–1472, 2013.
 [8] A. Kazeminejad and R. C. Sotero, “Topological properties of resting-state fMRI functional networks improve machine learning-based autism classification,” Frontiers in Neuroscience,vol. 12, p. 1018, 2019.
 [9]X. Li, N. C. Dvornek, J. Zhuang, P. Ventola, and J. S. Duncan, “Brain biomarker interpretation in asd using deep learning and fmri,” in Proceedings of the Medical Image Computing andComputer Assisted Intervention- MICCAI 2018, pp. 206–214, Springer, Granada, Spain,September 2018





