# Handwritten Text Recognition (HTR) system using Tensorflow.
Handwritten Text Recognition (HTR) system using Tensorflow and trained on the IAM off-line HTR dataset. This Neural Network (NN) model recognizes the text contained in the images of segmented words as shown in the illustration below.

![Example](https://github.com/githubharald/SimpleHTR/raw/master/doc/htr.png)
# Improvement from previous project [HTR](https://towardsdatascience.com/2326a3487cd5)
From the previous project, 5 layers CNN was used. But in my HTR, I will use 7 layers CNN in order to increase the accuracy.

![CNN 7 layers](https://github.com/BigBoss9797/HTR/blob/master/doc/cnn%207.JPG)

I also add word segmentation process, so that this system can recognize a sentence not just a word. Below is the example of sentence before segmentation.

![sentence before segmentation](https://github.com/BigBoss9797/HTR/blob/master/doc/a01-000u-s00-00.png)

After segmentation, the sentence split into 7 word images. 

![sentence after segmentation](https://github.com/BigBoss9797/HTR/blob/master/doc/segment.JPG)

# Train model (Google Colab)
[1] Download the IAM dataset from [here.](http://www.fki.inf.unibe.ch/DBs/iamDB/data/words)  
[2] Put the 'words' folder into \data directory.  
[3] Run using command line '!python src\Main.py --train' .  
[4] The complete model will be save in \model directory. (all files in this directory is the model, don't delete any of it.)  

# Word accuracy
The model achieve an accuracy of 76% which is improve from the previous project that I refer.

![Word Accuracy](https://github.com/BigBoss9797/HTR/blob/master/doc/wordAccuracy.JPG)

# Example Demo of HTR
I run my system using Flask. Below is the example.
First, choose image that we want to test.

![image](https://github.com/BigBoss9797/HTR/blob/master/doc/Screenshot%20(178).png)

Then, choose whether the image need contrast or not and click submit. The recognized text will showed up.

![result](https://github.com/BigBoss9797/HTR/blob/master/doc/Screenshot%20(177).png)

# References
[1] [Build a Handwritten Text Recognition System using Tensorflow](https://towardsdatascience.com/2326a3487cd5) 
