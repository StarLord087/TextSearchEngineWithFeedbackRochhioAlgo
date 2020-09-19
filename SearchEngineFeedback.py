#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nltk
import os
import string 
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from num2words import num2words
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import heapq
import re
import pandas as pd
from unidecode import unidecode
from sklearn.manifold import TSNE


# In[2]:


ps = PorterStemmer()


# In[3]:


path = 'C:\\Users\\shekh\\Desktop\\Shekhar\\IR\\Assignment 1\\20_newsgroups'
folders = []
files = []
docs = []
names = []

#loading folders
for folder in os.listdir(path):
    folders.append(folder)
    
#print(folders)

subset = ['comp.graphics', 'sci.med','talk.politics.misc', 'rec.sport.hockey', 'sci.space']

docsdic = {}

#loading files from each folder 
for folder in folders:
    if folder in subset:
        newpath = path +'\\'+folder
        for file in tqdm(os.listdir(newpath)):
            try:
                f = open(newpath+'\\'+file,"r")
                #print(doc)
                doc = f.read()
                #adding files to list
                docs.append(doc)
                if folder in docsdic:
                    docsdic[folder].append(file)
                else:
                    docsdic[folder] = [file]
                names.append((file,folder))
                f.close()
            except:
                pass
            
      


# In[4]:


#print(docsdic)


# In[5]:


def listtodict(keys, values):
    keys = keys.copy()
    values = values.copy()
    dic = {}
    for key in range(len(keys)): 
        for value in values: 
            dic[key] = value
            values.remove(value)
            break  
    return dic


# In[6]:


def remove_metadata(documents):
    pos = documents.index('\n\n')
    fixed_docs = documents[pos:]
    return fixed_docs


# In[7]:


def apply_proter_stemmer(string):
    for i in range(len(string)):
        string[i] = ps.stem(string[i])
    return string
        


# In[8]:


def remove_stopword(string):
    stop_words = set(stopwords.words('english'))
    data = [w for w in string if not w in stop_words]
    return data


# In[9]:


def convert_num(string):
    for i in range(len(string)):
        try:
            if(string[i].isnumeric()):
                string[i] = num2words(string[i])
        except:
            continue
    return string


# In[10]:


def remove_nonascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


# In[11]:


def preprocessing(newdataset):
    dataset = newdataset.copy()
    for i in tqdm(range(len(dataset))):
        dataset[i] = dataset[i].lower()
        dataset[i] = remove_nonascii(dataset[i])
        dataset[i] = dataset[i].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        dataset[i] = dataset[i].split()
        dataset[i] = remove_stopword(dataset[i])
        dataset[i] = convert_num(dataset[i])
        dataset[i] = apply_proter_stemmer(dataset[i])
    return dataset


# In[12]:


def querypreprocessing(stringdata):
    lower = stringdata.lower()
    punc = lower.translate(str.maketrans('','',string.punctuation))
    spl = punc.split()
    removestop = remove_stopword(spl)
    removenum = convert_num(removestop)
    stemmedquery = apply_proter_stemmer(removenum)
    
    for word in stemmedquery:
        if word not in vocab:
            stemmedquery.remove(word)
    
    return stemmedquery


# In[13]:


def create_vocabulary(data):
    vocab = set()
    for doc in data:
        for word in doc:
            #print(word)
            vocab.add(word)
    return list(vocab)


# In[14]:


def termfreq(data, word):
    if data.count(word) == 0:
        return 0.0
    else:
        return 1+(math.log10(data.count(word)))
        #return(data.count(word)/len(data))


# In[15]:


def create_dataframe(vocab, data):
    df = pd.DataFrame()
    df['vocab'] = vocab
    
    
    tfall = []
    
    for i in tqdm(range(len(data))):
        tf1 = []
        for word in vocab:
            tf1.append(termfreq(data[i],word))
        #print(len(tf1))
        tfall.append(tf1)
    #print(tfall)
    
    for i in tqdm(range(len(tfall))):
        df.insert(i+1, i, tfall[i],allow_duplicates = False)
    
    df.set_index('vocab', inplace = True)
    
    nonzero = np.count_nonzero(df,axis = 1) 
    
    #print("non zero",len(nonzero))
    
    idf = []
    for i in tqdm(range(len(nonzero))):
        #print(nonzero[i])
        idf.append(1/(1+(math.log10(df.shape[1]/nonzero[i]))))
    
    df['idf'] = idf
    return df


# In[16]:


def cosine_similarity(vec1, vec2):
    if np.isnan(np.dot(vec1, vec2)/(math.sqrt(np.sum(np.square(vec1)))*math.sqrt(np.sum(np.square(vec2))))):
        return 0
    else:
        return np.dot(vec1, vec2)/(math.sqrt(np.sum(np.square(vec1)))*math.sqrt(np.sum(np.square(vec2))))


# In[17]:


def create_tfidf(df1):
    df = df1.copy()
    df.iloc[:, 0:-1] = df.iloc[:, 0:-1].mul(df.iloc[:, -1], axis=0)
    #print(df)
    return df


# In[18]:


idtoname = listtodict(docs, names)


# In[19]:


#nametoid = listtodict(names, docs)


# In[20]:


print(idtoname[0])


# In[21]:


print(type(names[0][0]))


# In[22]:


nametoid = {}
for i in range(len(names)):
    nametoid[names[i][0]] = i


# In[23]:


print(type(nametoid['37261']))


# In[24]:


def cosine_retrieval(df1, querytfidf, vocabulary):
    
    df = df1.copy()
    
    
    #for i in querytfidf:
     #   if i > 0:
      #      print("queryvector",i)
    
    df = df.drop(labels = 'idf', axis = 1)
    #print(df.shape[1])
    #print(len(queryvector))
    similarity = []
    
    for i in range(df.shape[1]):
        val = df.iloc[:,i].values
        similarity.append(cosine_similarity(querytfidf, val))
    
    nonzero = []
    
    for i in similarity:
        if i > 0:
            nonzero.append(i)
    #print(nonzero)
    nonzero.sort(reverse=True)
    result = []
    #print(similarity)
    
    for i in nonzero:
        result.append(similarity.index(i))
        
    return result


# In[25]:


def create_query_vector(query, df1, vocabulary):
    df = df1.copy()
    
    processedquery = querypreprocessing(query)
    
    for word in processedquery:
        if word not in vocabulary:
            processedquery.remove(word)
    
    #print(processedquery.count("cindi")/len(processedquery))
    queryvector = np.zeros((df.shape[0]))
    #print(len(queryvector))
    wordvector = df.index.values 
    
    for i in range(len(wordvector)):
        for word in processedquery:
            if wordvector[i] == word:
                #print(wordvector[i], word)
                #print(processedquery.count(word)/len(processedquery))
                queryvector[i] = processedquery.count(word)/len(processedquery)
                #print(queryvector[i],i)
        
    
    idf = df.iloc[:,-1].values 
    #print("len of idf",len(idf))
    #print(queryvector)
    #print("len of query vector",len(queryvector))
    querytfidf = np.multiply(queryvector,idf)
    #print("length of query tfidf",len(querytfidf))
    
    return querytfidf
    


# In[26]:


for i in range(len(docs)):
    docs[i] = remove_metadata(docs[i])


# In[27]:


processed_data = preprocessing(docs)


# In[28]:


vocab = create_vocabulary(processed_data)


# In[29]:


print(len(vocab))


# Be patient it take about 10min

# In[30]:


inverted_index = create_dataframe(vocab, processed_data)


# ### Question 1
# the dataframe consists of all the tdidf values for each term for each document 

# In[32]:


tf_idf_invertedindex = create_tfidf(inverted_index)


# In[33]:


print(tf_idf_invertedindex.shape)


# In[34]:


#tf_idf_invertedindex.to_csv("tf_idf_invertedindex.csv", index = True)


# In[35]:


def rocchio_feedback(queryvec, reldocs, nonreldocs, df1):
    dataframe = df1.copy()
    modified_query_vector = 1*queryvec + (0.75*(dataframe[reldocs].sum(axis = 1)/len(reldocs))) - (0.25*(dataframe[nonreldocs].sum(axis = 1)/len(nonreldocs))) 
    
    for i in range(len(modified_query_vector)):
        if modified_query_vector[i] < 0:
            modified_query_vector[i] = 0
    
    return modified_query_vector


# In[36]:


def plot_pr(relevantfolder, alldata):
    
    retrievedres = []
    
    for i in range(len(alldata)):
        retrievedres.append(idtoname[alldata[i]][0])
    
    #print(retrievedres)
    
    relevant = docsdic[relevantfolder]
    #print(len(relevant))
    precision = []
    pmap = []
    recall = []
    retrieved = 0
    tp = 0
    

    for i in range(len(retrievedres)):
        if retrievedres[i] in relevant:
            tp += 1
        retrieved += 1
        #print(retrieved)
    
        precision.append(tp/retrieved)
        recall.append(tp/len(relevant))
    
    for i in range(len(retrievedres)):
        if retrievedres[i] in relevant:
            pmap.append(precision[i])
    
    avgpre = sum(pmap)
    MAP = avgpre/len(pmap)
    
    
    
    import matplotlib.pyplot as plt
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall curve')
    plt.plot(recall, precision)
    plt.show()
    
    return MAP
#print(relevant)


# In[37]:


def tsne_plot(reldocs, irreldocs, df1, query_vector):
    dataframe = df1.copy()
    
    labels = np.append(np.zeros(len(reldocs)),np.ones(len(irreldocs)))
    labels = np.append(labels, 2)
    labels =  labels.astype(int)    
    
    
    data = dataframe.loc[:,reldocs+irreldocs].values
    data = np.column_stack((data, query_vector))
    data = data.transpose()
    
    #print(data)
    
    tsne = np.array(TSNE(n_components = 2, random_state = 0).fit_transform(data))
    
    #print(tsne)
    
    colormap = np.array(['tab:blue', 'tab:orange', 'tab:green'])
    groups = np.array(["R", "N-R", "Query"])
    
    
    plt.scatter(tsne[:,0], tsne[:,1], alpha=0.8,s=50,c=colormap[labels], label = groups)
    plt.legend()
    plt.title("Rocchio Algorithm")
    plt.show()
    print("Blue: Relevant, Orange: Irrelevant, Green: Query")


# In[38]:


#tsne_plot(reldocs, irrdocs, tf_idf_invertedindex, query_vector)


# ### Question 2

# In[39]:



query = input("Enter Query: ")
k = int(input("Enter value of K: "))
feed_back_iteration = int(input("give number of iteration for feedback "))
query_vector = create_query_vector(query, tf_idf_invertedindex, vocab)
result_cosine = cosine_retrieval(tf_idf_invertedindex, query_vector, vocab)

kdocs = result_cosine[:k]
MAP = []
for i in range(int(k)):
        print(i,". ",idtoname[kdocs[i]])
        
for i in range(feed_back_iteration):
    print("Iteration: ",i+1)
    kdocs = result_cosine[:k]

    relevantinput = input("Enter index of relevant docs separated by space ")
    relevant_docs_input = relevantinput.split()

    relevant_docs_input = [int(i) for i in relevant_docs_input] 
        
    feedbackdf = tf_idf_invertedindex.loc[:,kdocs]

    reldocs = []
    irrdocs = []
    MAPi = [] 
    for i in relevant_docs_input:
        reldocs.append(kdocs[i])
    for i in kdocs:
        if i not in reldocs:
            irrdocs.append(i)

    query_vector = rocchio_feedback(query_vector,reldocs,irrdocs,feedbackdf)
    result_cosine = cosine_retrieval(tf_idf_invertedindex, query_vector, vocab)
    
    kdocs = result_cosine[:k]
    
    for i in range(int(k)):
        if kdocs[i] in reldocs:
            print(i,". ",idtoname[kdocs[i]], "*")
        else:
            print(i,". ",idtoname[kdocs[i]])

    mapi = plot_pr("sci.med",kdocs)
    MAPi.append(mapi)
    print("MAP: ",mapi)
    tsne_plot(reldocs, irrdocs, tf_idf_invertedindex, query_vector)
MAP.append(MAPi)


# In[40]:



query = input("Enter Query: ")
k = int(input("Enter value of K: "))
feed_back_iteration = int(input("give number of iteration for feedback "))
query_vector = create_query_vector(query, tf_idf_invertedindex, vocab)
result_cosine = cosine_retrieval(tf_idf_invertedindex, query_vector, vocab)

kdocs = result_cosine[:k]
MAP = []
for i in range(int(k)):
        print(i,". ",idtoname[kdocs[i]])
        
for i in range(feed_back_iteration):
    print("Iteration: ",i+1)
    kdocs = result_cosine[:k]

    relevantinput = input("Enter index of relevant docs separated by space ")
    relevant_docs_input = relevantinput.split()

    relevant_docs_input = [int(i) for i in relevant_docs_input] 
        
    feedbackdf = tf_idf_invertedindex.loc[:,kdocs]

    reldocs = []
    irrdocs = []
    MAPi = [] 
    for i in relevant_docs_input:
        reldocs.append(kdocs[i])
    for i in kdocs:
        if i not in reldocs:
            irrdocs.append(i)

    query_vector = rocchio_feedback(query_vector,reldocs,irrdocs,feedbackdf)
    result_cosine = cosine_retrieval(tf_idf_invertedindex, query_vector, vocab)
    
    kdocs = result_cosine[:k]
    
    for i in range(int(k)):
        if kdocs[i] in reldocs:
            print(i,". ",idtoname[kdocs[i]], "*")
        else:
            print(i,". ",idtoname[kdocs[i]])

    mapi = plot_pr("talk.politics.misc",kdocs)
    MAPi.append(mapi)
    print("MAP: ",mapi)
    tsne_plot(reldocs, irrdocs, tf_idf_invertedindex, query_vector)
MAP.append(MAPi)


# In[51]:



query = input("Enter Query: ")
k = int(input("Enter value of K: "))
feed_back_iteration = int(input("give number of iteration for feedback "))
query_vector = create_query_vector(query, tf_idf_invertedindex, vocab)
result_cosine = cosine_retrieval(tf_idf_invertedindex, query_vector, vocab)

kdocs = result_cosine[:k]
MAP = []
for i in range(int(k)):
        print(i,". ",idtoname[kdocs[i]])
        
for i in range(feed_back_iteration):
    print("Iteration: ",i+1)
    kdocs = result_cosine[:k]

    relevantinput = input("Enter index of relevant docs separated by space ")
    relevant_docs_input = relevantinput.split()

    relevant_docs_input = [int(i) for i in relevant_docs_input] 
        
    feedbackdf = tf_idf_invertedindex.loc[:,kdocs]

    reldocs = []
    irrdocs = []
    MAPi = [] 
    for i in relevant_docs_input:
        reldocs.append(kdocs[i])
    for i in kdocs:
        if i not in reldocs:
            irrdocs.append(i)

    query_vector = rocchio_feedback(query_vector,reldocs,irrdocs,feedbackdf)
    result_cosine = cosine_retrieval(tf_idf_invertedindex, query_vector, vocab)
    
    kdocs = result_cosine[:k]
    
    for i in range(int(k)):
        if kdocs[i] in reldocs:
            print(i,". ",idtoname[kdocs[i]], "*")
        else:
            print(i,". ",idtoname[kdocs[i]])

    mapi = plot_pr("sci.med",kdocs)
    MAPi.append(mapi)
    print("MAP: ",mapi)
    tsne_plot(reldocs, irrdocs, tf_idf_invertedindex, query_vector)
MAP.append(MAPi)

