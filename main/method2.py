
# coding: utf-8

# In[259]:


from gensim.models import word2vec
from gensim import models
from pprint import pprint
from scipy import spatial
import numpy as np
import time
import os
import json
import csv


# In[260]:


def loadModel(path):
    model = models.Word2Vec.load(path)
    print("Success load model!")
    return model


# In[261]:


def readData(path):
    t = time.time()
    with open(path, 'r') as reader:
        data = json.loads(reader.read())
    print("It took %.2f sec to read data" % (time.time() - t))
    print(data)
    return data


# In[262]:


# ==================
#       method 2
# ==================
def generateAnswer(data):
    C_con = np.zeros(250, dtype = float)
    QA_con = np.zeros((250, 250), dtype = float)
    ca = data['correct_answer']
    anslist = ['A', 'B', 'C', 'D']
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

    for j in range(0, 4):
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
    
    tag = (anslist[ans] == ca )
    print("The predict answer is %s." %(anslist[ans]))
    print("The correct answer is %s." %ca)
    return tag


# In[263]:


def main():
    t = time.time()
    pathModel = '../word2vec/wiki/python/word2vec.model'
    pathData = '../CQA/CQA_'
    totalData = 394
    count = 0
    tagList = False 
    
#======  read data in for loop  ======
    for i in range(0, 394):
        print("Start reading data in" + pathData + str(i) + '.json')
        jsonData = readData(pathData + str(i) + '.json')
        
        print("Start generate output of" + pathData + str(i) + '.json')
        ansTag = generateAnswer(jsonData)
        
        if ansTag == True:
            count +=1
    accuracy = (count/totalData)*100
    print("=========Finished========")
    print("The accuracy is %.2f percent" % accuracy )
    print("It took %.2f sec to process" % (time.time() - t))
    
    output_title = "wiki_avg_accuracy"
    
#====== output data =======
    print("Output......")
    
    with open("method2_20180610.txt", 'a+') as file:
        file.write("\n")
        file.write("gigaword(cowb600)_avg_accuracy")
        file.write("\n")
        file.write(str(accuracy))

    print("Output done!")


# In[264]:


pathModel = '../word2vec/giga/gigaword(cowb600)/word2vec.model'
model = models.Word2Vec.load(pathModel)
print("Success load model!")

if __name__ == "__main__":
    main()

