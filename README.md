# Disease Prediction
Disease Prediction 
As we witness the Health Care industry they have been Digitally transformed, life scientists and doctors have an ocean of data to base their research upon and Additionally huge volumes of health-related information are made accessible through many widely spread adoption of wearable tech. This opens up new opportunities for better, more informed healthcare. We are able to collect, structure and process a high volume of data and further make sense of it, to gain a much more deeper understanding of the human body its key objectives.it also has the strongest potential to revolutionize healthcare. This Data can be leveraged and used for a great transformation in Health care –  
As we see the Health care especially over the  PHC- level where the Government PHC’s (Primary health care centres) are structured in a way that a lot of people below poverty line and Low income and  economic level visit the PHC for their diagnose and health check-up but due to 1.Lack of Availability of Doctors  2.Unavailability of Doctors on times of patient need 3.Lack of knowledge of  environmental conditions 4. No proper efficient Diagnose for certain Diseases. To bridge this gap, we are bringing DISEASETAP - Disease prediction and Diagnose Automation through using Data Science and Machine learning where, we build Huge Datasets which include patient Lipid profiles and Blood test samples reports etc. we build Different Models for each disease and run a prediction analysis over these datasets with Careful and Prior knowledge over the parameters prescribed by the doctors in Diagnosing these diseases and then they are inculcated into our Programming interfaces, made into a running Prediction Models and Classifiers which will intern predict the disease the patient is effected with and also his Risk Factor

2. Software System Design
2.1.1 Python language: Python is a general-purpose language; it has the right tools/libraries. 
 
Fig 2.1.1: Logo of the python language

2.1.2 Anaconda navigator: we are using Anaconda Navigator as a desktop graphical user interface (GUI) included in Anaconda® we are launching applications and using the conda packages, environments, and channels without using command-line commands.

2.1.3 Juypter Notebook: we are using the Jupyter Notebook in Anaconda Navigator, we are creating and sharing documents that contain live code, equations, visualizations and explanatory text. Uses include: data cleaning and transformation, numerical simulation, statistical modelling, machine learning and much more.
 
      Fig 2.1.2: Anaconda navigator and we have to enable Juypter notebook


2.1.4 Libraries needed for the implementation of Code:
1.	numpy: To work with arrays
2.	pandas: To work with csv files and dataframes
3.	matplotlib: To create charts using pyplot, define parameters using rcParams and color them with cm.rainbow
4.	warnings: To ignore all warnings which might be showing up in the notebook due to past/future depreciation of a feature
5.	train_test_split: To split the dataset into training and testing data
6.	StandardScaler: To scale all the features, so that the Machine Learning model better adapts to the dataset

2.1.5 Import dataset
After downloading the dataset from Kaggle, saved it to the working directory with the name dataset.csv.

2.1.6 Create a New Environment in Anaconda Navigator:  
Fig 2.1.3: This figure shows bellow the two steps first to go into environments and then click on create.
Fig 2.1.4: Enter the name of the environment and also select the language and its version
2.1.7 Now in the new environment installing the above stated libraries / packages (Mentioned in 2.1.4)
Fig 2.1.5: Type the Package name in the search bar and click apply to install (same way install all the needed libraries and packages as mentioned in 2.1.4) 

3. IMPLEMENTATION

We are importing the data set from Kaggle saved it to the working directory with the name dataset.csv. Next we used describe() method  Which reveals that the range of each variable is different. The maximum value of age is 77 but for chol it is 564. Thus, feature scaling must be performed on the dataset.
Understanding the data - Correlation Matrix
It enables us to see the correlation matrix of features and also try to analyse it. The figure size is defined to 12 x 8 by using rcParams. Then, used pyplot to show the correlation matrix. Using xticks and yticks, added names to the correlation matrix. colorbar() shows the colorbar for the matrix.

Here we can see that there is no single feature that has a very high correlation with our target value. Also, some of the features have a negative correlation with the target value and some have positive.
Histogram: The best part about this type of plot is that it just takes a single command to draw the plots and it provides so much information in return. Just use dataset.hist().

If we take a closer look at these plots. It shows how each feature and label is distributed along different ranges, which further confirms the need for scaling. wherever we see discrete bars, it basically means that each of these is actually a categorical variable. We will need to handle these categorical variables before applying Machine Learning. Our target labels have two classes, 0 for no disease and 1 for disease.
Bar Plot for Target Class
It’s essential that the dataset we are working on should be approximately balanced. An extremely imbalanced dataset can render the whole model training useless and thus, will be of no use, so we plotting a Target class Bar plot to see how balanced the dataset are. 
For x-axis we use the unique() values from the target column and then set their name using xticks. For y-axis, we use value_count() to get the values for each class. colored the bars as green and red.

Data Processing
To work with categorical variables, we should break each categorical column into dummy columns with 1s and 0s. Let’s say we have a column Gender, with values 1 for Male and 0 for Female. It needs to be converted into two columns with the value 1 where the column would be true and 0 where it will be false.

# Original Columm # Dummy Columns                         
# |  Gender_0  ||  Gender_1  |
# |      0     ||      1     |
# |      0     ||      1     |
# |      1     ||      0     | 

In order to do this, we use the get_dummies() method from pandas. 

 we need to scale the dataset for which we will use the StandardScaler. The fit_transform() method of the scaler scales the data and we update the columns.
Now once the DATASET is ready, we can dive into the next part.
Machine Learning
In this project, we took 4 algorithms and varied their various parameters and compared the final models. I split the dataset into 67% training data and 33% testing data.
We use 4 different classifiers
K Neighbors Classifier
Support Vector Classifier
Decision Tree Classifier
Random Forest Classifier
K Neighbors Classifier
This classifier looks for the classes of K nearest neighbors of a given data point and based on the majority class, it assigns a class to this data point. However, the number of neighbors can be varied. we varied them from 1 to 20 neighbors and calculated the test score in each case.
knn_scores = []
	for k in range(1,21):
	    knn_classifier = KNeighborsClassifier(n_neighbors = k)
	    knn_classifier.fit(X_train, y_train)
	    knn_scores.append(knn_classifier.score(X_test, y_test))
Code - Manasseh John Wesley 

ploting the  line graph of the number of neighbors and the test score achieved in each case.











plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
	for i in range(1,21):
	    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
	plt.xticks([i for i in range(1, 21)])
	plt.xlabel('Number of Neighbors (K)')
	plt.ylabel('Scores')
	plt.title('K Neighbors Classifier scores for different K values')

Decision Tree Classifier: This classifier creates a decision tree based on which, it assigns the class values to each data point. Here, we can vary the maximum number of features to be considered while creating the model. we range features from 1 to 30 (the total features in the dataset after dummy columns were added).



Random Forest Classifier: This classifier takes the concept of decision trees to the next level. It creates a forest of trees where each tree is formed by a random selection of features from the total features. Here, we can vary the number of trees that will be used to predict the class. we calculate test scores over 10, 100, 200, 500 and 1000 trees
We plot these scores across a bar graph to see which gave the best results. we do not directly set the X values as the array [10, 100, 200, 500, 1000]. It will show a continuous plot from 10 to 1000, which would be impossible to decipher. So, to solve this issue, we first use the X values as [1, 2, 3, 4, 5]. Then, I renamed them using xticks.


1.	K Neighbors Classifier: 87%
2.	Support Vector Classifier: 83%
3.	Decision Tree Classifier: 79%
4.	Random Forest Classifier: 84%

Conclusion AND FURTHER USE:
The analysis of the disease patient dataset with proper data processing. 4 models were trained and tested with maximum scores as follows.
We use this technology in wide range of Diseases to predict who are affected and their risk factor and we can cover a large number of patients at the same time with fast and efficient analysis.
By using this we can improve patient care, chronic disease management and increasing the efficiency of supply chains and pharmaceutical logistics. Population health management is becoming an increasingly popular topic in predictive analytics. This is a data-driven approach focusing on prevention of diseases that are commonly prevalent in society.
With Diseasetap, hospitals can predict the deterioration in patient’s health and provide preventive measures and start an early treatment that will assist in reducing the risk of the further aggravation of patient health. Furthermore, predictive analytics plays an important role in monitoring the logistic supply of hospitals and pharmaceutical departments.
This has many applications in healthcare. The medicine and healthcare industry have heavily utilized Data Science for the improving lifestyle of patients and predicting diseases at an early stage. Furthermore, with advancements in medical image analysis, it is possible for the doctors to find out microscopic tumors that were otherwise hard to find. Therefore, data science has revolutionized healthcare and the medical industry in large ways.
