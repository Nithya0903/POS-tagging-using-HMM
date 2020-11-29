from collections import OrderedDict
import re
from math import isnan,isinf
from copy import deepcopy
from random import random
import preprocess

def initialize_pi(tags):
    initial_prob = dict()
    total = 0.0
    for tag in tags:
        initial_prob[tag] = random()
        total += initial_prob[tag]

    initial_prob = dict((tag,(initial_prob[tag])/total) for tag in tags)
    return initial_prob

def calc_gamma(alpha, beta, tags, observation):
    for i, word in enumerate(observation):
        if word == '':
            del observation[i]
        else:
            1

    gamma = dict()
    for i in range(1, len(observation)+1):
        gamma[i] = dict()

    for i in range(1, len(observation)+1):
        for tag in tags:
            gamma[i][tag] = (beta[i][tag] * alpha[i][tag]) / alpha[len(observation)+1]
            # print("ALPHA", alpha[i][tag] , beta[i][tag] , alpha[len(observation)+1])

    return gamma

def calc_eta(a_matrix, b_matrix, alpha, beta, tags, observation):
    for i, word in enumerate(observation):
        if word == '':
            del observation[i]
        else:
            1
    eta = dict()
    for i in range(1, len(observation)+1):
        eta[i] = dict()
        for tag in tags:
            eta[i][tag] = dict()
    
    for i in range(1, len(observation)):
        for tag1 in tags:
            for tag2 in tags:
                # print(alpha[len(observation)+1])
                # if alpha[len(observation)+1]==0:
                #     print(alpha)
                eta[i][tag1][tag2] = (alpha[i][tag1] * a_matrix[tag1][tag2] * beta[i+1][tag2] * b_matrix[tag2][observation[i]])/alpha[len(observation)+1]
    return eta

def initialize_a(tags):
    a = dict()
    for i in tags:
        for j in tags:
            if i not in a:
                a[i] = dict()
            a[i][j] = random()
        a[i]['f'] = random()
    return a
#changes made here
def initialize_b(tags, line_list):
    b={}
    for tag in tags:
        b[tag]={}

    for sentence in line_list:
        for tag in tags:
            sum = 0.
            for word in sentence:
                if word=='':
                    continue
                b[tag][word]= random()
                sum += b[tag][word]
            for word in sentence:
                if word=='':
                    continue
                b[tag][word] = b[tag][word]/sum
        
        

    return b

def normalize_a(a, tags):
    a_matrix = deepcopy(a)
    total = 0.0
    for tag1 in tags:
        for tag2 in tags:
            total = total + a_matrix[tag1][tag2] 
        total = total + a_matrix[tag1]['f']
    # if total==0:
    #     print("problem")
    #     exit()
    for tag1 in tags:
        for tag2 in tags:
            p = a_matrix[tag1][tag2]
            a_matrix[tag1][tag2] = (a_matrix[tag1][tag2])/total
            # if isnan(a_matrix[tag1][tag2]):
            #     print("problem")
            #     print(p,total)
            #     exit()
        a_matrix[tag1]['f'] = (a_matrix[tag1]['f'])/total
        #print("normalisied A:", tag1, a_matrix[tag1]['f'])

    return a_matrix

# changes made here
def inlayer_norm_b(b, tag_list, observation):
    total = dict()
    # for  tag in tag_list:
    #     print(tag,b[tag]['reelection'])
    for word in observation:
        if word == '':
            continue
        else:
            for i in tag_list:
                if i not in total:
                    total[i] = 0
                total[i] += b[i][word]
                # if isnan(total[i]):
                #     print(i,b[i][word],word)
                #     exit()
    for i in tag_list:
        for word in observation:
            # if total[i]==0:
            #     print("problem")
            #     exit()
            p= b[i][word]
            b[i][word] = (b[i][word])/total[i]
            # if isnan(b[i][word]):
            #     print("problem")
            #     print(p,total[i])
            #     print(total,i)
            #     exit()

    return b

#changes made here
# def normalize_b(b, tags, sentences):
#     for sentence in sentences:
#         total = dict()
#         for word in sentence:
#             if word == '':
#                 continue
#             else:
#                 for tag in tags:
#                     if tag not in total:
#                         total[tag] = 0.0
#                     total[tag] += b[tag][word]
#         for tag in tags:
#             for word in sentence:
#                 if word =='':
#                     continue 
#                 else:   
#                     b[tag][word] = (b[tag][word])/total[tag]
#     for tag in tags:
#         b[tag]={}

#     for sentence in line_list:
#         for tag in tags:
#             sum = 0.
#             for word in sentence:
#                 b[tag][word]= random()
#                 sum += b[tag][word]
#             for word in sentence:
#                 b[tag][word] = b[tag][word]/sum
#     return b

def backward(a_matrix, b_matrix, pi, scale_values, observation, tags):       # beta[timestamp][tag]

    beta = dict()
    for i, word in enumerate(observation):
        if word == '':
            del observation[i]
        else:
            1
    # print(observation)

    for i in range(1,len(observation)+1):
        beta[i] = dict()
    # Initialize the T timestamp probs
    for tag in tags:
        # beta[len(observation)][tag] = a_matrix[tag]['f']
        beta[len(observation)][tag] = scale_values[len(observation)]

    for i in range(len(observation)-1, 0, -1):
        j = i+1
        for tag_pres in tags:
            beta[i][tag_pres] = 0.0
            for tag_future in tags:  
                # print(type(tag_future),tag_future,type(observation[i]),observation[i])
                # print(b_matrix[tag_future][observation[i]]) 
                beta[i][tag_pres] += (beta[j][tag_future] * a_matrix[tag_pres][tag_future] * b_matrix[tag_future][observation[i]])
            beta[i][tag_pres] = scale_values[i] * beta[i][tag_pres]
    #Final layer computation
    final = 0
    beta[final] = 0.0
    for tag in tags:
        beta[final] += (beta[final+1][tag] * pi[tag])
    return beta

# def pos_tags():
#     tags = ['NP', 'NN', 'JJ', 'IN', 'VB', 'TO', 'DT', 'PRP', 'RB', 'CC']
#     return tags

#change
def forward(a_matrix, b_matrix, pi, observation, tags):        # alpha[timestamp][tag]
    alpha = dict()
    for i, word in enumerate(observation):
        if word == '':
            del observation[i]
        else:
            1
    scale_values = dict()

    for i in range(1,len(observation)+1):
        alpha[i] = dict()
    # initialize
    alpha[1] = dict((tag,(pi[tag]*b_matrix[tag][observation[0]])) for tag in tags)
    
    #scaling for the first timestep
    scale_values[1] = 0.0
    for tag in tags:
        scale_values[1] += alpha[1][tag]

    scale_values[1] = 1.0/scale_values[1]
    alpha[1] = dict((tag,(alpha[1][tag]*scale_values[1])) for tag in tags)

    for i in range(2, len(observation)+1):
        j = i-1
        scale_values[i] = 0
        for tag_pres in tags:
            alpha[i][tag_pres] = 0.0
            for tag_prev in tags: 
                alpha[i][tag_pres] += (alpha[j][tag_prev] *  b_matrix[tag_pres][observation[j]] * a_matrix[tag_prev][tag_pres])

            scale_values[i] += alpha[i][tag_pres]
        
        scale_values[i] = 1.0/scale_values[i]
        for tag in tags:
            alpha[i][tag] = scale_values[i] * alpha[i][tag]
            # if alpha[i][tag]==0:
            #     print("PR")
            #     exit()
    #Final layer computation
    final = len(observation)+1
    alpha[final] = 0.0
    for tag in tags:
        alpha[final] += (a_matrix[tag]['f']*alpha[final-1][tag])
        # print("alpha[x]",x,tag,alpha[final-1][tag],a_matrix[tag]['f'],alpha[final])
    return alpha, scale_values

def baum_welch(a_matrix, b_matrix, tags, line_list):
    # observation = line_list[0]
    for k,observation in enumerate(line_list):
        # if k == 20:
        #     break
        if observation == '':
            continue
        else:
            print("Iteration number:",k)
            print("Sentence number:",observation)
        
        for i, word in enumerate(observation):
            if word == '':
                del observation[i]
            else:
                1
        
        for i in range(10):     # Fixed number of observations = 1000
            print("-----------",i,"------------")
            #E-STEP
            alpha, scale_values = forward(a_matrix, b_matrix, pi, observation, tags)
            beta = backward(a_matrix, b_matrix, pi, scale_values, observation, tags)
            # alpha = forward(a_matrix, b_matrix, pi, observation, tags)
            
            eta = calc_eta(a_matrix, b_matrix, alpha, beta, tags, observation)
            gamma = calc_gamma(alpha, beta, tags, observation)
            
            #M-STEP
            prev_b_matrix = deepcopy(b_matrix) 
            prev_a_matrix = deepcopy(a_matrix)

            for tag in tags:
                for word in observation:
                    numer = denom = 0.0
                    for t in range(1, len(observation)):
                        if observation[t] == word:
                            numer += gamma[t][tag]
                        else:
                            numer += 0
                        denom += gamma[t][tag]
                # if isinf(numer) or numer>1e306:
                #     numer=1e306
                # if isinf(denom) or denom>1e306:
                #     denom=1e306
                b_matrix[tag][word]  = (numer)/denom


            for tag1 in tags:
                for tag2 in tags:
                    numer = denom = 0.0
                    for t in range(1, len(observation)):
                        numer += eta[t][tag1][tag2]
                        for temp_tag in tags:
                            denom += eta[t][tag1][temp_tag]
                    # if denom==0:
                    #     print("Problemmmmm")
                    # if isinf(numer) or numer>1e306:
                    #     numer=1e306
                    # if isinf(denom) or denom>1e306:
                    #     denom=1e306

                    a_matrix[tag1][tag2] = (numer)/denom


            # b_matrix = inlayer_norm_b(b_matrix, tags, observation)
            # a_matrix = normalize_a(a_matrix, tags)
    return a_matrix, b_matrix

def tokenize(line_list):
    lines = []
    regex = re.compile('[\W]+')

    for line in line_list:
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
    return lines            

def calc_acc(b_matrix,words,labels,tags):
    print(len(words),len(labels),len(tags))
    #use only words which have one of the tags as labels
    from sklearn.metrics import accuracy_score
    selected_words =[]
    truelabels = []
    # selected_words =words
    # truelabels = labels
    #print(tags)
    for i,label in enumerate(labels):
        if label in tags:
            selected_words.append(words[i])
            truelabels.append(label)
        else:
            print("{} not in tags".format(label))
            
    predicted_labels = []

    for w in selected_words:
        maxx=0
        max_tag=''
        for tag in tags :
            try:
                if b_matrix[tag][w]>maxx :
                    maxx=b_matrix[tag][w]
                    max_tag=tag
            except:
                max_tag='-'
        predicted_labels.append(max_tag)
    print("Predicting for ",len(selected_words))
    n = len(truelabels)
    c =0
    for i in range(n):
        if truelabels[i] == predicted_labels[i]:
            c+=1

    score =(c*100)/n
    #score = accuracy_score(truelabels,predicted_labels)
    print(c,score)




if __name__ == '__main__':
    words,labels,tags,text = preprocess.preprocessor()
    line_list = tokenize(text)
    print(len(words),len(labels),len(tags))
    print("There are {} tags".format(len(tags)))
    #tags=tags[:20]
    #tags = ['np', 'nn', 'jj', 'in', 'vb', 'to', 'dt', 'prp', 'rb', 'cc']
    #print(line_list[:10])
    #print(line_list)
    print("Initializing initial probability.....")
    pi = initialize_pi(tags)
    print("Initializing a tags.....")
    a_matrix = initialize_a(tags)
    print("Initializing b tags.....")
    b_matrix = initialize_b(tags, line_list)
    #print(pi)
    #a_matrix = normalize_a(a_matrix, tags)
    #b_matrix = normalize_b(b_matrix, tags, line_list)
    # print b_matrix['NN']['fulton']
    
    a_matrix, b_matrix = baum_welch(a_matrix, b_matrix, tags, line_list)
    # for dic in b_matrix['NP']:
    #     ordered_b = OrderedDict(sorted(b_matrix.iteritems(), key=lambda x: x[1], reverse=True))
    #print ( ordered_b )
    top_dict = {}
    for tag in b_matrix:
        col = b_matrix[tag]
        top = sorted(col, key=col.get, reverse=True)
        top = top[:100]
        top_dict[tag] = top

    with open('out-comp.txt','w+') as f:
        for key in top_dict:
            words1 = ', '.join(top_dict[key])
            f.write(key + ': '+ words1 + '\n')
    
    with open('a-matrix-comp.txt','w+') as f1:
        for i in (a_matrix):
                f1.write("{} {}".format(i,a_matrix[i]))
        f1.write("\n")
    with open('b-matrix-comp.txt','w+') as f2:
        for i in (b_matrix):
            for j in b_matrix[i]:
                f2.write("{} {} {}".format(i,j,b_matrix[i][j]))
                f2.write("\n")
            f2.write("\n")
    # print("A- MATRIX")
    # print(a_matrix)
    # print("\n\n\n")
    # print(b_matrix)
    print("Calculating accuracy....")
    print(len(words),len(labels),len(tags))
    calc_acc(b_matrix,words,labels,tags)