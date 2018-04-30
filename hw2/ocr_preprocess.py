import re
import numpy as np

N_EXAMPLES = 5000
N_FEATURES = 128

def atoi(a):
    return int(ord(a)-ord('a'))
def itoa(i):
    return chr(i+ord('a'))

def iors(s):
    try:
        return int(s)
    except ValueError: # if it is a string, return a string
        return s

def read_OCR(filename, n_examples = N_EXAMPLES, n_features = N_FEATURES):
    F = open(filename)
    dataset = {}
    dataset['ids'] = np.zeros(n_examples, dtype=int)
    dataset['labels'] = np.zeros(n_examples,dtype=int)
    dataset['next_ids'] = np.zeros(n_examples,dtype=int)
    dataset['word_ids'] = np.zeros(n_examples,dtype=int)
    dataset['positions'] = np.zeros(n_examples,dtype=int)
    dataset['folds'] = np.zeros(n_examples,dtype=int)
    dataset['features'] = np.zeros([n_examples,n_features])
    
    i = 0
    for str_line in F.readlines():
        line0 = list(map(iors, filter(None, re.split('\t', str_line.strip()))))
        
        dataset['ids'][i] = line0.pop(0)
        dataset['labels'][i] = atoi(line0.pop(0))
        dataset['next_ids'][i] = line0.pop(0)
        dataset['word_ids'][i] = line0.pop(0)
        dataset['positions'][i] = line0.pop(0)
        dataset['folds'][i] = line0.pop(0)
        if len(line0) != 128:  # Sanity check of the length
            print(len(line0))

        for j, v in enumerate(line0):
            dataset['features'][i][j] = v
        i += 1
        if i == n_examples:
            break
            
    return dataset

def chop_idxs(ocr, window = 2, start = 0, stop = None):
    if stop is None: stop = len(ocr['ids'])
    chops = []
    chop = []
    i = start
    while i < stop:
        nextid = ocr['next_ids'][i]
        if len(chop) < window:
            chop.append(i)
            if nextid == -1 or i == stop-1:
                while len(chop) < window:
                    chop.append('_')
                if i == stop-1:
                    chops.append(chop)
            i = i+1
        else:
            chops.append(chop)
            chop = []
    return(np.array(chops))

def chops_to_str(ocr, chops):
    return np.array([[itoa(ocr['labels'][int(idx)]) if idx != '_' else '_' for idx in chop] for chop in chops])

def chops_to_labels(ocr, chops):
    return np.array([[ocr['labels'][int(idx)] if idx != '_' else 26 for idx in chop] for chop in chops])

def chops_to_features(ocr, chops):
    return np.array([[ocr['features'][int(idx)] if idx != '_' else np.zeros(N_FEATURES) for idx in chop] for chop in chops])