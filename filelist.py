import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt


'''
we're using  python3 for this . . 
'''

datain="043b2gg_pre.dat"
datainr="043b2gg.dat"
path='./xxdmini/mapbr/es/'
import os
rootdir = '.'


class File_Compile:
	def __init__(self, rootdir):
		self.datain= rootdir

	def lister(self):
		filels=[]
		for subdir, dirs, files in os.walk(rootdir):
			for file in files:
				if (pd.Series(file).str.contains('pre')[0]) and (pd.Series(file).str.contains('dat')[0]):
					filels.append(str(os.path.join(subdir, file)))

				
		fls=pd.Series(filels)
		filels=fls[fls.str.contains('/es/')]
		return filels

b=File_Compile(rootdir)
filels=b.lister()
