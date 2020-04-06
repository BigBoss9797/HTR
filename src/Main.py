from __future__ import division
from __future__ import print_function
 
import sys
import argparse
import cv2
import editdistance
import os
import numpy as np
import tensorflow as tf
import shutil

from ImageLoader import ImageLoader, Batch
from Model import Model
from ImagePreprocessor import preprocess
from WordSegmentation import wordSegmentation, prepareImg
 

 
class FilePaths:
    "filenames and paths to data"
    fnCharList = 'SimpleHTR/model/charList.txt'
    fnAccuracy = 'SimpleHTR/model/accuracy.txt'
    fnTrain = 'SimpleHTR/data/'
    fnDerive = 'SimpleHTR/data/1.png'
    fnCorpus = 'SimpleHTR/data/corpus.txt'
 
 
def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 25 # stop training after this number of epochs without improvement
    batchNum = 0

    while True:
        epoch += 1
        print('Epoch:', epoch)
 
        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            batchNum += 1
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch, batchNum)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)
 
        # validate
        charErrorRate,wordAccuracy = validate(model, loader)

        cer_summary = tf.Summary(value=[tf.Summary.Value(
            tag='charErrorRate', simple_value=charErrorRate)])  # Tensorboard: Track charErrorRate
        # Tensorboard: Add cer_summary to writer
        model.writer.add_summary(cer_summary, epoch)

        accuracy_summary = tf.Summary(value=[tf.Summary.Value(
            tag='wordAccuracy', simple_value=wordAccuracy)])  # Tensorboard: Track addressAccuracy
        # Tensorboard: Add address_summary to writer
        model.writer.add_summary(accuracy_summary, epoch)
 
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%% \nWord accuracy of saved model: %f%%' % (charErrorRate*100.0, wordAccuracy*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1
 
        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break
 
 
def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)
 
        print('Ground truth -> Recognized')    
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
 
    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate, wordAccuracy
 
 
def derive(model, fnImg, contrast=False):
    "recognize text in image provided by file path"
 
    img = prepareImg(cv2.imread(fnImg), 50)
    res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)

    if os.path.exists('out/'):
      shutil.rmtree('out')

    if os.path.exists('cont/'):
      shutil.rmtree('cont')

    if not os.path.exists('out/'):
      os.mkdir('out/')

    if not os.path.exists('cont/'):
      os.mkdir('cont/')
      
    print('Segmented into %d words'%len(res))
    for (j, w) in enumerate(res):
      (wordBox, wordImg) = w
      (x, y, w, h) = wordBox
      cv2.imwrite('out/%d.png'%j, wordImg) 
      cv2.rectangle(img,(x,y),(x+w,y+h),0,1) 
 
    if contrast==True:
      imgFiles = sorted(os.listdir('out/'))
      word =""
      for (i,f) in enumerate(imgFiles):
          img = cv2.imread('out/%s'%f, cv2.IMREAD_GRAYSCALE)
  
          # increase contrast
          pxmin = np.min(img)
          pxmax = np.max(img)
          imgContrast = (img - pxmin) / (pxmax - pxmin) * 255
  
          # increase line width
          kernel = np.ones((3, 3), np.uint8)
          imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)
          cv2.imwrite('cont/%d.png'%i, imgMorph)
          imgClean = sorted(os.listdir('cont/'))
      for (m,n) in enumerate(imgClean):
          imgC = cv2.imread('cont/%s'%n, cv2.IMREAD_GRAYSCALE)
          img = preprocess(imgC, Model.imgSize)
          batch = Batch(None, [img])
          (recognized, probability) = model.inferBatch(batch, True)
          print('Recognized:', '"' + recognized[0] + '"')
          print('Probability:', probability[0])
          word += recognized[0] + " "
      print(word)
    else:
      imgFiles = sorted(os.listdir('out/'))
      word =""
      for (m,n) in enumerate(imgFiles):
          imgC = cv2.imread('out/%s'%n, cv2.IMREAD_GRAYSCALE)
          img = preprocess(imgC, Model.imgSize)
          batch = Batch(None, [img])
          (recognized, probability) = model.inferBatch(batch, True)
          print('Recognized:', '"' + recognized[0] + '"')
          print('Probability:', probability[0])
          word += recognized[0] + " "
      print(word)
    return word

 
def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
 
    args = parser.parse_args()
 
    # train or validate on IAM dataset    
    if args.train or args.validate:
        # load training data, create TF model
        loader = ImageLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
 
        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
 
        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
 
        # execute training or validation
        if args.train:
            model = Model(loader.charList)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, mustRestore=True)
            validate(model, loader)
 
    # infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), mustRestore=True)
        derive(model, FilePaths.fnDerive, contrast=False)
        #derive(model, FilePaths.fnDerive, contrast=True)


def derive_by_web(path, option):
    if option=='require contrast':
      print(open(FilePaths.fnAccuracy).read())
      model = Model(open(FilePaths.fnCharList).read())
      recognized = derive(model, path, contrast=True)

      return recognized 
    elif option=='no need contrast':
      print(open(FilePaths.fnAccuracy).read())
      model = Model(open(FilePaths.fnCharList).read())
      recognized = derive(model, path, contrast=False)

      return recognized 

if __name__ == '__main__':
    main()