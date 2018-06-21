
# coding: utf-8

# In[91]:


from gensim.models import word2vec
from gensim import models
from pprint import pprint
from scipy import spatial
import numpy as np
import time
import os
import json


# In[92]:


def loadModel(path):
    model = models.Word2Vec.load(path)
    print("Success load model!")
    return model


# In[93]:


def readData(path):
    t = time.time()
    with open(path, 'r') as reader:
        data = json.loads(reader.read())
    print("It took %.2f sec to read data" % (time.time() - t))
    return data


# In[94]:


# ==================
#    method 1
# ==================
def generateAnswer(data):
    CQ_con = np.zeros(250, dtype = float)
    A_con = np.zeros((250, 250), dtype = float)
    ca = data['correct_answer']
    anslist = ['A', 'B', 'C', 'D']
    CQ_list = data['corpus']
    CQ_list.extend(data['question'])

    for word in CQ_list:
        try:
            vector = model[word]
        except KeyError as e:
            continue
        for i in range(250):
            CQ_con[i] += vector[i]
            
#     for i in range(250):
#         CQ_con[i] /= 250

    for j in range(0, 4):
        for word in data['answer'][j]:
            try:
                vector = model[word]
            except KeyError as e:
                continue
            for i in range(250):
                A_con[j][i] += vector[i]
#             for i in range(250):
#                 A_con[j][i] /= 250


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
    
    tag = (anslist[ans] == ca )
    print("The predict answer is %s." %(anslist[ans]))
    print("The correct answer is %s." %ca)
    return tag


# In[95]:


def  main():
    t = time.time()
    pathMode = '../word2vec/giga/python/word2vec.model'
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

#====== output data =======
    output_title = "wiki_avg_accuracy"
    
    print("Output......")
    
    with open("method1_20180619.txt", 'a+') as file:
        file.write("\n")
        file.write("wiki(skip300)_sum_accuracy")
        file.write("\n")
        file.write(str(accuracy))

    print("Output done!")


# In[96]:


pathModel = '../word2vec/wiki/wiki_zh_tw(skip300)/word2vec.model'
model = models.Word2Vec.load(pathModel)
print("Success load model!")

if __name__ == "__main__":
    main()

