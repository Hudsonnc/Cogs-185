import numpy as np
from pystruct.datasets import load_letters

def splitWord(feature, label, window):
    feature = np.array(feature)
    label = np.array(label).reshape(-1,1)
    
    num_letters = feature.shape[0]
    letter_features = feature.shape[1]
    
    feature_splits = []
    label_splits = []
    if num_letters < window:
        features_to_append = np.zeros((window - num_letters, letter_features))
        labels_to_append = np.array([26 for i in range(window - num_letters)]).reshape((-1,1))
        
        feature_split = (np.vstack((feature, features_to_append)))
        label_split = (np.vstack((label, labels_to_append)))
        
        feature_splits.append(feature_split)
        label_splits.append(label_split)
    else: 
        for i in range(num_letters - window + 1):
            feature_split = feature[i:(i+window)]
            label_split = label[i:(i+window)]
            
            feature_splits.append(feature_split)
            label_splits.append(label_split)
    return(np.array(feature_splits), np.array(label_splits))  

def splitWords(features, labels, window):
    feature_splits, label_splits = zip(*list(map(lambda x, y: splitWord(x,y,window), features, labels)))
    feature_splits = np.vstack(feature_splits)
    label_splits = np.vstack(label_splits)
    return(feature_splits, label_splits)

def selectWordsByLetters(words, start, stop):
    i = 0
    newWords = []
    for word in words:
        newWord = []
        for letter in word:
            if i >= start:
                if i < stop:
                    newWord.append(letter)
                else:
                    if newWord:
                        newWords.append(np.array(newWord))
                    return(np.array(newWords))
            i = i + 1
        if newWord:
            newWords.append(np.array(newWord))
            
def loadWindows(start, stop, window):
    letters = load_letters()
    X, y = letters['data'], letters['labels']
    X, y = np.array(X), np.array(y)
    word_features = selectWordsByLetters(X, start, stop)
    word_labels = selectWordsByLetters(y, start, stop)
    window_features, window_labels = splitWords(word_features, word_labels, window)
    return(window_features.astype(np.double), window_labels.astype(np.double))
    