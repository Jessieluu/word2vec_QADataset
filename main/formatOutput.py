# -*- coding: UTF-8 -*-
import time
import jieba
import json
import os
import shutil

# call jieba api
def jiebaCut(s):
    # 精確模式 ：將句子最精確地切開，叫適合文本分析, cut_all=False
    # 全模式：把句子中所有的可以成詞的詞語都掃描出來, 速度快, cut_all=True
    # 搜索引擎模式：在精確模式的基礎上對長詞再次切分，提高召回率，適合用於搜尋引擎分詞, jieba.cut_for_search(Content)
    words = jieba.cut_for_search(s)
    return words

# output result file
def outputFile(cList, qList, aList, ind):
    content = {}
    fileName = 'CQA/' + '/' + str(ind) + '.json'
    with open(fileName, 'w') as CQA:
        content["corpus"] = cList
        content["question"] = qList
        content["answer"] = aList
        CQA.write(json.dumps(content, ensure_ascii=False))

# main func start point
def main():
    sTime = time.time()
    print("Start process CQA dataset")
    ind, wrongind = 0, 0
    filePath = os.getcwd() + '/Data/'
    # start process CQA data set
    for f in range(30000):
        if os.path.exists(filePath + str(f) + '.txt'):
            with open(filePath + str(f) + '.txt', 'r') as file:
                print("Process file " + str(f))
                flag = 0
                cList, qList, aList = [], [], []
                state = {'C': 1, 'Q': 2, 'A': 3}
                for i in file.readlines():
                    s = i.strip()
                    if state.get(s) is not None:
                        flag = state.get(s)
                        continue
                    if flag == 1:
                        cutRes = jiebaCut(s)
                        for c in cutRes:
                            cList.append(c)
                    elif flag == 2:
                        cutRes = jiebaCut(s)
                        for c in cutRes:
                            qList.append(c)
                    elif flag == 3:
                        # example: 一 三 二號 兩艘 三 五艘 四 二十三艘
                        temp = {'一', '二', '三', '四'}
                        tempS = ""
                        check = 0
                        for i in range(len(s)):
                            if s[i] == " ":
                                continue
                            elif s[i] in temp and check == 0:
                                check = 1
                            elif s[i] in temp and check == 1 and tempS != "":
                                tempL = []
                                cutRes = jiebaCut(tempS)
                                for c in cutRes:
                                    tempL.append(c)
                                aList.append(tempL)
                                tempS = ""
                            elif check == 1:
                                tempS += s[i]

                        if tempS != "":
                            tempL = []
                            cutRes = jiebaCut(tempS)
                            for c in cutRes:
                                tempL.append(c)
                            aList.append(tempL)
                        # if answer is wrong format, then output none for json
                        # and write this question number to txt
                        if len(aList) == 4:
                            outputFile(cList, qList, aList, ind)
                        else:
                            cList, qList, aList = [], [], []
                            outputFile(cList, qList, aList, ind)
                            wrongind += 1
                            with open("wrongNum.txt", 'a') as file:
                                file.write(str(f))
                                file.write("\n")

                        cList, qList, aList = [],[],[]
                        flag = 0
                        ind += 1
                        
    print("\nTotal corpus numbers: %d" % ind)
    print("\nWrong answer of corpus numbers: %d" % wrongind)
    print("Processing all CQA dataset corpus took %.2fs" % (time.time()- sTime))


# ====== initial setting ======

print("Start loading initial setting!")
# jieba setting
print("Start loading jieba dictionary!")
relativePath = os.getcwd()
jieba.set_dictionary(relativePath + '/jieba_setting/dict.txt.big')
# add user dictionary to improve jieba cut precision
# jieba.load_userdict(relativePath + '/jieba_setting/yourfile.txt')

# ====== initial setting ======


if __name__ == "__main__":
    main()
