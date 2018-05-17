import os
import numpy as np

folder_path='.'

filenames = os.listdir(folder_path)  # get all files' and folders' names in the current directory

result = []
subfoldertarget='scores'
scores={}
std={}
mean_scores={}
mean_std={}
for filename in filenames:  # loop through all the files and folders
    if os.path.isdir(os.path.join(os.path.abspath("."), filename)):# check whether the current object is a folder or not
        try:
            subfolder=os.listdir(os.path.join(filename, subfoldertarget))
        except:
            continue
        #os.path.join(filename, subfoldertarget)
        scores = {filename: []}
        std = {filename: []}
        mean_scores = {filename: []}
        mean_std = {filename: []}
        for file in subfolder:
            text_file = open(os.path.join(filename, subfoldertarget,file), "r")
            lines = text_file.readlines()

            for i_line in range(0,len(lines)):
                if lines[i_line].split('=')[0]=='Mean score':
                    scores[filename].append(float(lines[i_line].split('=')[1][0:5]))
                if lines[i_line].split('=')[0]=='Std. dev.':
                    std[filename].append(float(lines[i_line].split('=')[1][0:5]))
        print(filename)
        print('mean_score')
        print(np.mean(scores[filename]))
        print('mean_std')
        print(np.std(scores[filename]))
        print('\n')
        #mean_scores[filename].append(np.mean(scores[filename]))
        #mean_std[filename].append(np.std(scores[filename]))
#print(filename)
#print('mean_score')
#print(mean_scores)
#print('std')
#print(mean_std)
        #result.append(filename)

