import re
def preprocessor():
    import pandas as pd
    import numpy as np

    data = pd.read_csv("datasets/brown.csv")
    data = data[:139]
    print(data.tail())
    tkntext = data["tokenized_text"]
    #print(tkntext.head())
    pos= data["tokenized_pos"]
    wordlist = []
    truetags=list()
    sentencelist= np.array(tkntext)
    tokenllist= np.array(pos)
    tags=[]
    words=[]
    uniquetags = []
    for tknline in tokenllist:
        t = tknline.split(" ")
        for tag in t:
            tags.append(tag)
    

    for  i,sentence in enumerate(sentencelist):
        s = sentence.split(" ")
        for word in s:
            words.append(word)
    
    # x = dict(zip(words,tags))
    # from collections import Counter
    # coun = Counter(x)
    # words = []
    # tags = []
    # for word, tag in coun.most_common(500): 
    #     words.append(word)
    #     tags.append(tag)
    uniquetags = list(set(tags))
 
    tkntext= np.array(tkntext)
    print(len(words),len(tags),len(uniquetags))

    return (words,tags,uniquetags,tkntext)
preprocessor()

def tokenize(filename):
    lines = []
    regex = re.compile('[\W]+')
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(' ')
            if line[0].startswith("#"):
                continue
            else:
                for i, word in enumerate(line):
                    line[i] = re.sub('[^a-zA-Z0-9]+', '', line[i])
                    if line[i] == '':
                        del line[i]
                    else:
                        line[i] = line[i].lower()

                if not line:
                    continue
                lines.append(line)
    print(lines)
    return lines            
#tokenize("datasets/brown.csv")