import pandas as pd
import numpy as np
from PIL import Image
import os, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import time
import random
from tqdm.notebook import tqdm
from sklearn.linear_model import Lasso, LassoLars
from sklearn.preprocessing import LabelEncoder

os.chdir(r'd:/Murgi/code/memes2024/meme-research-2024')

meme_df = pd.read_parquet('./data/meme_entries.parquet')
template_df = pd.read_parquet('./data/meme_template_links.parquet')

labelencoder = LabelEncoder()
meme_df['template_id'] = labelencoder.fit_transform(meme_df['template_name'])
meme_df['template_id'] = meme_df['template_id'] + 1
sample_size = 30
sampled_meme_df = meme_df.groupby('template_name').apply(lambda x: x.sample(n=min(sample_size, len(x)),random_state=42))
sampled_meme_df = sampled_meme_df.reset_index(drop=True)
sampled_meme_df

def stack(X_train,X_test):
    X_toconcat_train = [np.reshape(e,(X_train[0].shape[0]*X_train[0].shape[1],1)) for e in X_train]
    X_toconcat_test = [np.reshape(e,(X_train[0].shape[0]*X_train[0].shape[1],1)) for e in X_test]
    
    Xtrain = np.concatenate(X_toconcat_train,axis=1) # Each column is now an image of the train set
    Xtest = np.concatenate(X_toconcat_test,axis=1) # Each column is now an image of the test set
    return Xtrain, Xtest

def stack2(X_train):
    X_toconcat_train = [np.reshape(e,(X_train[0].shape[0]*X_train[0].shape[1],1)) for e in X_train]
    Xtrain = np.concatenate(X_toconcat_train,axis=1) # Each column is now an image of the train set
    return Xtrain
    
def single_stack(image):
    # Check if 'image' is a list; if not, wrap it in a list
    if not isinstance(image, list):
        image = [image]
    # Reshape and stack as before
    X_toconcat = [np.reshape(e, (e.shape[0] * e.shape[1], 1)) for e in image]
    return np.concatenate(X_toconcat, axis=1)  # Each column is now an image

def read_images(meme_df, target_size=(64,64), filter=Image.LANCZOS):
    '''
    Read the images from the meme_df dataframe and return the images and the labels
    - meme_df: dataframe containing the path to the images and the template_id
    - target_size: size of the output images (default is (128,128))
    - filter: filter to use for resizing (default is Image.LANCZOS)
    '''
    path_list = meme_df['path'].tolist()
    template_list = meme_df['template_id'].tolist()
    X,y = [], []
    for (path, template) in tqdm(zip(path_list, template_list), total= len(path_list)):
        try:
            im = Image.open(path) 
            im = im.convert("L")
            # resize to given size (if given) and check that it's the good size
            im = im.resize(target_size, resample=filter)
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(template)
        except IOError:
            pass
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
    print("Images uploaded !")
    return np.array(X), np.array(y)

def read_image(path, target_size=(64,64), filter=Image.NEAREST):
    im = Image.open(path) 
    im = im.convert("L")
    # resize to given size (if given) and check that it's the good size
    im = im.resize(target_size, resample=filter)
    return np.asarray(im, dtype=np.uint8)

def residual_vectorized(y, A, x, class_x):
    '''
    Returns the residuals of the model for each class
    n: number of features, m: number of samples
    - y: the target vector, shape (n,)
    - A: training images, shape (n, m)
    - x: the coefficients, shape (m,)
    - class_x: the train labels, shape: (m,) type: list
    '''
    # Generate the delta matrix for all classes
    delta_matrix = delta_vectorized(x, class_x)
    
    # Compute the predicted values for each class
    predictions = np.dot(A, delta_matrix)
    
    # Calculate the errors by subtracting y from each column of predictions
    # Note: Need to reshape y to broadcast correctly against predictions
    errors = predictions - y.reshape(-1, 1)
    
    # Calculate the norm of errors for each class to get the residuals
    r = np.linalg.norm(errors, axis=0)
    
    return r

def delta_vectorized(x, class_num):
    '''
    Function that selects the coefficients associated with the ith class
    Useful for SCI calculation
    m: number of samples
    - x: vector of coefficients, shape (m,)
    - class_num: vector of training labels, shape (m,)
    '''
    n, m = len(x), len(class_num)
    
    if (n != m):
        print('Vectors of different sizes')

    k = np.max(class_num)+1

    tmp = np.subtract(np.multiply(np.ones((n, k)),np.arange(k)),class_num[:, np.newaxis])
    tmp = np.where(tmp == 0, 1, 0)
    
    return (tmp * x[:, np.newaxis])

import pickle as pkl

save_path_X = "./data/X_original_64x64_sampled_30.pkl"
save_path_y = "./data/y_original_64x64_sampled_30.pkl"
load_df = sampled_meme_df

if os.path.exists(save_path_X):
    print("Load")
    X_original = pkl.load(open(save_path_X, 'rb'))
    y_original = pkl.load(open(save_path_y, 'rb'))

else:
    # X_original, y_original = read_images(load_df, filter=Image.NEAREST)
    with open(save_path_X, 'wb') as f:
        pkl.dump(X_original, f)

    with open(save_path_y, 'wb') as f:
        pkl.dump(y_original, f)

fb_df = pd.read_parquet(r"data\touch\facebook.parquet")
twitter_df = pd.read_parquet(r"data\touch\twitter.parquet")
reddit_df = pd.read_parquet(r"data\touch\reddit.parquet")

fb_df=fb_df.rename(columns={"filepath":"path"})
fb_df.loc[:,"path"] =  fb_df.loc[:,"path"].apply(lambda x: x.replace("D:\\Facebook2023","D:\\Murgi\\Facebook2023\\Facebook2023"))
fb_df

twitter_df=twitter_df.rename(columns={"filepath":"path"})
twitter_df.loc[:,"path"] =  twitter_df.loc[:,"path"].apply(lambda x: x.replace("D:\\Twitter2023","D:\\Murgi\\Twitter2023"))
twitter_df

reddit_dir = 'C:\\Users\\molontay\\Murgi\\data\\Memes2022Final2\\Memes2022Final2\\' 
files = os.listdir(reddit_dir)
id_to_files = {os.path.splitext(file)[0]:file for file in files}

reddit_df['path'] = reddit_df['id'].apply(lambda x: os.path.join(reddit_dir,id_to_files[x]) if id_to_files[x] else None)

import random

random.seed(123)

all_reddit_memes = reddit_df['path'].tolist()
sample_reddit_memes = random.sample(all_reddit_memes, 1000)
rest_reddit_memes = list(set(all_reddit_memes) - set(sample_reddit_memes))

all_fb_memes = fb_df['path'].tolist()
sample_fb_memes = random.sample(all_fb_memes, 1000)
rest_fb_memes = list(set(all_reddit_memes) - set(sample_fb_memes))


all_twitter_memes = twitter_df['path'].tolist()
sample_twitter_memes = random.sample(all_twitter_memes, 1000)
rest_twitter_memes = list(set(all_twitter_memes) - set(sample_twitter_memes))

all_the_rest = rest_reddit_memes + rest_fb_memes + rest_twitter_memes
random.shuffle(all_the_rest)


X_paths = sample_reddit_memes + sample_fb_memes + sample_twitter_memes + all_the_rest

from sklearn.preprocessing import StandardScaler
from tqdm.contrib.concurrent import  thread_map, process_map  # or thread_map


X_stacked = stack2(X_original)
ss = StandardScaler()
ss.fit(X_stacked.T)
X_scaled = ss.transform(X_stacked.T).T

def predict(path:str,X_scaled,y_original,ss:StandardScaler):
    img = read_image(path)
    img_stacked = single_stack(img)
    img_scaled = ss.transform(img_stacked.T).T

    clf = LassoLars(alpha=0.35, max_iter=1000)
    clf.fit(X_scaled,img_scaled)
    x = clf.coef_
    pred = np.argmin(residual_vectorized(img_scaled, X_scaled, x, y_original))
    with open("sparse_matching_sm_preds.txt","a") as f:
        f.write(f"{path}\t{pred}\n")
    return (path,pred)


print("Setup finished. Starting inference:")
# Run inference on X_paths using thread_map
results = thread_map(lambda path: predict(path, X_scaled, y_original, ss), X_paths)

