import os
import pandas as pd
from pathlib import Path

inFile = 'oversampled.csv'
overdf = pd.read_csv(inFile)
overlist = overdf.iloc[:, 1].tolist()

for filename in overlist:
    fileid = filename.replace("Train_original\\", "")
    inPath = Path('Train_original').joinpath(fileid)
    outPath = Path('Train').joinpath(fileid)
    outFile = filename.replace("Train_original", "Train")
    os.system("C:\\Users\\mel-s\\ffmpeg -i {0} -ar 48000 {1}".format(inPath, outPath))
