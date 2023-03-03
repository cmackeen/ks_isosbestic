import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import scipy as scp
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.integrate import simps
from scipy.optimize import curve_fit

'''
we're using  python3 for this . .
'''
a=.002
#datain="043b2gg_pre.dat"
#datals=['br_110Kff.dat','br_145Kff.dat','br_160Kff.dat','br_185Kff.dat','br_200Kff.dat','br_232Kff.dat','br_243Kff.dat','br_275Kff.dat','br_300Kff.dat']
datals=['br_10Kff.dat','br_70Kff.dat','br_110Kff.dat','br_145Kff.dat','br_160Kff.dat','br_185Kff.dat','br_200Kff.dat','br_232Kff.dat','br_243Kff.dat','br_275Kff.dat','br_300Kff.dat']

fk_datain='f_keep_br-pb_mapbr.dat'
epsilon=.000003
scaleisobk=[6.459,7.10805,7.70214,8.28475,8.84835,9.40328,9.98276]
#for pp in range(len(scaleisobk)):
#	if pp%2==0:
#		scaleisobk[pp]=.997*scaleisobk[pp]
#	else:
#		scaleisobk[pp]=.997*scaleisobk[pp]


scaleisobchi=[.0188722,-.02005,.022369,-.02211,.0226168,-.0206913,.0117305]
temps=[10,70,110,145,160,185,200,232,243,275,300]
class KData:
	temps=[10,70,110,145,160,185,200,232,243,275,300]
	def __init__(self, datarg):
		self.datals= datarg

	def reader(self):
		kz_cross=[]
		plt.figure()

		with open(fk_datain) as myfile:
				head = [next(myfile) for x in range(25)]
				hd_len=pd.Series(head).str.contains('#').sum()

				# creating interpolation k-vector



		for datain in self.datals:
			smoothdf=[]
			smoothdf_scaled=[]

			df=[]


			with open(datain) as myfile:
				head = [next(myfile) for x in range(25)]
				hd_len=pd.Series(head).str.contains('#').sum()
			df=pd.read_csv(datain, skiprows=hd_len, sep='\s+')
			df.columns=['k','mu','mmm']

			interpx=np.linspace(4.1,df['k'].max(),8000)

			dffk=pd.read_csv(fk_datain, skiprows=hd_len, sep='\s+')
			dffk.columns=['k','mu']
			int_funcfk=interp1d(dffk['k'],dffk['mu'], kind='cubic')
			smoothdffk=[]
			for j in interpx:
				smoothdffk.append(int_funcfk(j))




			int_func=interp1d(df['k'],df['mu'], kind='cubic')
			#int_funcz=np.polyfit(df['k'],df['mu'], deg=20)

			for j in interpx:
				smoothdf.append(int_func(j))
				smoothdf_scaled.append(int_func(j)/int_funcfk(j))

			zeros=[]

			for pp in range(len(interpx)-1):
				if interpx[pp]>6.2:
					if smoothdf[pp]*smoothdf[pp+1]<0:
						#print('eriuhnrejcn')
						zeros.append(interpx[pp])


			#print(kz_cross)
			kz_cross.append(zeros)



			#plt.plot(interpx,smoothdf,label= str(datain).split('.')[0].split('_')[1].split('ff')[0],marker='o')
			plt.plot(interpx,smoothdf_scaled,label= 'scaled'+str(datain).split('.')[0].split('_')[1].split('ff')[0],marker='*')
			#plt.plot(outt['k'],outt['mu'],label=str(datain))
			plt.legend(fontsize=16)
			plt.title('Back FF Br-Pb k-space')
			plt.xlabel('k ($\AA^{-1}$)')
			plt.ylabel('k*$\chi$ (k)')

			plt.show()
			odf = pd.concat([pd.Series(interpx),pd.Series(smoothdf_scaled)], axis=1)
			odf.to_csv(str(datain)+'fkdiv', sep='\t')




		return df, kz_cross, int_func, smoothdf_scaled,interpx



go=KData(datals)
outt=go.reader()
smoo=outt[3]
interp=outt[4]
allzs=pd.DataFrame(outt[1])
############### SIgs fit reader
'''
sigs=pd.read_csv('br_sigsqrs.dat', sep='\s')
sigs=sigs.dropna(axis=1)
sigs.columns=['t','sig','unc']

linear_regressor = LinearRegression()  # create object for the class
bou=linear_regressor.fit(sigs['t'], sigs['sig'])

sigslope=bou.coef_[0]
drdt=(1/290.)*(3.0017-2.9816) # perform linear regression
sig_pred = linear_regressor.predict(np.linspace(20,400,200).reshape(-1,1))  # make predictions
plt.figure()
plt.plot(sigs['t'],sigs['sig'], marker='o')
plt.plot(np.linspace(20,400,200),sig_pred)
'''
#drdt=(1/290.)*(3.0017-2.9816)
drdt=0
sigslope=4.61*10**-5
######################################


##### c3 fit reader
c3s=pd.read_csv('br_c3fit.dat', sep='\s+')
c3s.columns=['t','c3','unc']
c3s['c33']=c3s['c3']**3
c3s['c3dif']=c3s['c33'].diff()
c3s['tdif']=c3s['t'].diff()
#plt.figure()
#c3s['slope']=c3s['c3dif']/c3s['tdif']
#c3s.drop([5,6,7,8,9],axis=0,inplace=True)
#plt.plot(c3s['t'],c3s['c33'])
#plt.plot(c3s['t'],c3s['slope'])

###############################

allzsb=allzs.drop([10,11],axis=1)
rpb=[2.993599,2.994199,2.9947,2.9957,2.9988,3.0016,3.002,3.0074,3.0067,3.0058999,3.0116]
rbrfix=[2.9816,2.9832,2.9843,2.9884,2.9905,2.993,2.993,2.998,2.997,2.9991,3.0017]
rbrfree=[2.9833,2.985,2.98715,2.9809,2.980067,2.984417,2.97768,2.9672,2.96803,2.969,2.9639]
r0=2.993599
k0=allzsb.ix[0]
del_matrix=pd.DataFrame().reindex_like(allzsb)
D_matrix=pd.DataFrame().reindex_like(allzsb)
ratmat=pd.DataFrame().reindex_like(allzsb)
kbratmat=pd.DataFrame().reindex_like(allzsb)
## jj is n (nth zerocross)
# hh is i (ith temp)
for jj in range(len(allzsb.ix[0])-4):
	for hh in range(len(allzsb[0])):
		del_matrix[jj][hh]=allzsb[jj][hh]-scaleisobk[jj]
		if jj>1 and jj<len(allzsb.ix[0])-4:
			D_matrix[jj][hh]=0.5*(allzsb[jj+1][0]-allzsb[jj-1][0])
			ratmat[jj][hh]=del_matrix[jj][hh]/D_matrix[jj][hh]
			kbratmat[jj][hh]=1.*(3/2.)*(1/scaleisobk[jj]**2)*(drdt+scaleisobk[jj]*sigslope*np.pi*del_matrix[jj][hh]/D_matrix[jj][hh])

		print(allzsb[jj][hh]-allzsb[jj][0])


kbratmat.dropna(axis=1,inplace=True)
t_avg_ratmat=kbratmat.mean(axis=1)
#t_avg_ratmat=t_avg_ratmat.apply(lambda x: x-(t_avg_ratmat[0]))
#t_avg_ratmat=pd.concat([pd.Series([0]), t_avg_ratmat], ignore_index=True)
#outrat=t_avg_ratmat.apply(inplace=True, lambda x: -1.*(x))
temps2=[0,10,70,110,145,160,185,200,232,243,275,300]
c3xtr=[]
for t in range(len(temps)):
	c3xtr.append(simps(y=t_avg_ratmat[0:t+1],x=temps[0:t+1]))
c3plot_offset=pd.Series(c3xtr)
c3plot_offset=c3plot_offset.apply(lambda x: x - (c3plot_offset[1]-1.49779e-05))
plt.figure()
plt.plot(pd.Series(temps[1:]),c3plot_offset[1:],label='extarct', marker='*')

plt.plot(c3s['t'],c3s['c33'])

c3plot_offset.to_csv('isob_br_c3.dat', sep='\t')

plt.legend()

'''
plt.figure()

for i in range(len(ratmat.columns)):
	plt.plot(temps, ratmat[ratmat.columns[i]],label=ratmat.columns[i])

plt.legend()
'''