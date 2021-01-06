NLP - Classifiers (Bag Of & Numeric)

This project was created by: George Skoufias, Marousopoulou Maria, Konstantinos Vrailas

1. From the folder covid19-tweetCorpus unzip the file corona.zip and create a copy to the NLP - Classifiers (Bag Of & Numeric) folder.

2. Run as administrator setup.bat. This batch file is safe to run. Its purpose is to create a virtual environment in the current directory. Subsequently, it runs inside this environment a series of pip install commands based on a dependency descriptor file – requirements.txt – that is also included in the folder. In this manner the transferred project becomes of minimum size. If run as administrator does not work, navigate inside the project folder using command prompt. Run setup.bat from there.

3.	To run profiling of twitter_samples issue command: python TweetSamplesProfile.py. The standard output is directed in two files inside the reports directory called positive_tweets_report.txt, negative_tweets_report.txt.

4.	To run profiling of corona issue command: python CovidProfile.py. The standard output is directed in two files inside the reports directory called covid_report.txt, covid_tagged_report.txt.

5.	To run all the classifiers based on the labeled feature sets, run: python TrainCalssifier.py. To alter POS filtering use/substitute any of the values 0,1,2,3 in lines 69, 165, 170.

6.	To run the classifiers based on occurrence counts and the BoX approach run: python numeric_classifier.py.
