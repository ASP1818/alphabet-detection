import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X, y = fetch_openml("mnist_784", version = 1, return_X_y = True)
print(pd.Series(y).value_count())
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "V", "X", "Y", "Z"]
nclasses = len(classes)

if(not os.environ.get("PYTHONHTTPSVERIFY", '')and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml("mnist_784", version = 1, return_X_y = True)
print(pd.Series(y).value_count())
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nclasses = len(classes)