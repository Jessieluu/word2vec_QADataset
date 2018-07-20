
# coding: utf-8

from gensim.models import word2vec
from gensim import models
from pprint import pprint
from scipy import spatial
import numpy as np
import time
import os
import json
import csv
import random


def readData(path):
    t = time.time()
    with open(path, 'r') as reader:
        data = json.loads(reader.read())
    #print("It took %.2f sec to read data" % (time.time() - t))
    return data


# ==================
#       method 2
# ==================
def generateAnswer(data):
    C_con = np.zeros(250, dtype = float)
    QA_con = np.zeros((250, 250), dtype = float)
    #ca = data['correct_answer']
    anslist = ['1', '2', '3', '4']
    C_list = data['corpus']
    QA_list = []               
    
    for j in range (0, 4):           
        QA_list.append(data['question'])      
        
    for word in C_list:
        try:
            vector = model[word]
        except KeyError as e:
            continue
        for i in range(250):
            C_con[i] += vector[i]
    
    for i in range(250):
        C_con[i] /= 250

    for j in range(0, len(data['answer'])): 
        QA_list[j].extend(data['answer'][j])
        for word in QA_list[j]:
            try:
                vector = model[word]
            except KeyError as e:
                continue
            for i in range(250):
                QA_con[j][i] += vector[i]
            for i in range(250):
                QA_con[j][i] /= 250

    ini = 0
    high_cq = 0
    i = 0
    ans = 0
    
    for qa in QA_con:
        cos = 1 - spatial.distance.cosine(C_con, qa)
        if cos > ini:
            ini = cos
            high = qa
            ans = i
        i += 1
    
    #tag = (anslist[ans] == ca )
    #print("The predict answer is %s." %(anslist[ans]))
    #print("The correct answer is %s." %ca)
    return anslist[ans]



def main():
    t = time.time()
    pathData = './Result/'
    # clear result
    f = open('method2_result.txt', 'w')
    f.close()
    totalData = 1500
    wrongid = 0
    ansList = []
#======  read data in for loop  ======
    for i in range(totalData):
        #print("Start reading data in" + pathData + str(i) + '.json')
        jsonData = readData(pathData + str(i) + '.json')
        
        #print("Start generate output of" + pathData + str(i) + '.json')

#=== check format is correct or not ===
        randomNum = True
        if len(jsonData['answer']) != 4:
            randomNum = False
            wrongid += 1
        if randomNum == False:
            ansTag = str(random.randint(1,4))
        elif randomNum == True:
            ansTag = generateAnswer(jsonData)

        ansList.append(ansTag)
        
#====== output data =======
        # with open("method2_result.txt", 'a+') as file:
        #     file.write(ansTag)
        #     file.write("\n")

    outputList = []
    outputMerge = []
    readFileName = 'output.csv'
    outputFileName = 'outputMerge.csv'

    #read question number from csv file
    with open(readFileName, newline= '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        outputList = list(spamreader)

        for i in range(0, len(ansList), 1):
            data = str(outputList[i+1][0])+ str(ansList[i])
            outputMerge.append(data)

    with open(outputFileName, 'w', newline='') as csvfile:
        csvfile.write('ID,Answer')
        csvfile.write('\n')
        for i in outputMerge:                     
            csvfile.write(i)    
            csvfile.write('\n')
        
        #print("Output done!")
        
    print("=========Finished========")
    print("Total wrong corpus format numbers are %d" % wrongid)
    print("It took %.2f sec to process" % (time.time() - t))
    # print(ansList)

pathModel = './word2vec/wiki/wiki_zh_tw(skip300)/word2vec.model'
model = models.Word2Vec.load(pathModel)
print("Success load model!")

if __name__ == "__main__":
    main()

