
# coding: utf-8

from gensim.models import word2vec
from gensim import models
from pprint import pprint
from scipy import spatial
import numpy as np
import time
import os
import json
import random


def readData(path):
    t = time.time()
    with open(path, 'r') as reader:
        data = json.loads(reader.read())
   #print("It took %.2f sec to read data" % (time.time() - t))
    return data


# ==================
#    method 1
# ==================
def generateAnswer(data):
    CQ_con = np.zeros(250, dtype = float)
    A_con = np.zeros((250, 250), dtype = float)
    #ca = data['correct_answer']
    anslist = ['1', '2', '3', '4']
    CQ_list = data['corpus']
    CQ_list.extend(data['question'])

    for word in CQ_list:
        try:
            vector = model[word]
        except KeyError as e:
            continue
        for i in range(250):
            CQ_con[i] += vector[i]
            
    for i in range(250):
        CQ_con[i] /= 250

    for j in range(0, 4):
        for word in data['answer'][j]:
            try:
                vector = model[word]
            except KeyError as e:
                continue
            for i in range(250):
                A_con[j][i] += vector[i]
            for i in range(250):
                A_con[j][i] /= 250
    ini = 0
    high_cq = 0
    i = 0
    ans = 0
    
    for a in A_con:
        cos = 1 - spatial.distance.cosine(a, CQ_con)
        if cos > ini:
            ini = cos
            high = a
            ans = i
        i += 1

    print("The predict answer is %s." %(anslist[ans]))
    return anslist[ans]


def main():
    t = time.time()
    pathData = './CQA/'
    totalData = 1500
    
#======  read data in for loop  ======
    for i in range(totalData):
        print("Start reading data in" + pathData + str(i) + '.json')
        jsonData = readData(pathData + str(i) + '.json')
        
        print("Start generate output of" + pathData + str(i) + '.json')

#=== check format is correct or not ===
        randomNum = True
        if len(jsonData['answer']) != 4:
            randomNum = False
        if randomNum == False:
            ansTag = str(random.randint(1,4))
        elif randomNum == True:
            ansTag = generateAnswer(jsonData)

#====== output data =======
        
        with open("method1_result.txt", 'a+') as file:
            file.write(ansTag)
            file.write("\n")

        print("Output done!")
        
    print("=========Finished========")
    print("It took %.2f sec to process" % (time.time() - t))


pathModel = './word2vec/wiki/wiki_zh_tw(cowb300)/word2vec.model'
model = models.Word2Vec.load(pathModel)
print("Success load model!")

if __name__ == "__main__":
    main()

