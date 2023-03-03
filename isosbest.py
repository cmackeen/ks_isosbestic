import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import scipy as scp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

'''
we're using  python3 for this . .
'''
a=.002
#datain="043b2gg_pre.dat"
datals=['br_10Kff.dat','br_70Kff.dat','br_110Kff.dat','br_145Kff.dat','br_160Kff.dat','br_185Kff.dat','br_200Kff.dat','br_232Kff.dat','br_243Kff.dat','br_275Kff.dat','br_300Kff.dat']
#path used for testing
path='./ks/'
sects=[[8.3,8.4],[9.4,9.5],[]]
epsilon=.000003
temps=[10,70,110,145,160,185,200,232,243,275,300]
class KData:
	temps=[10,70,110,145,160,185,200,232,243,275,300]
	def __init__(self, datarg):
		self.datals= datarg

	def reader(self):
		kz_cross=[]
		plt.figure()
		for datain in self.datals:
			smoothdf=[]
			df=[]


			with open(datain) as myfile:
				head = [next(myfile) for x in range(25)]
				hd_len=pd.Series(head).str.contains('#').sum()
			df=pd.read_csv(datain, skiprows=hd_len, sep='\s+')
			df.columns=['k','mu','mmm']
			interpx=np.linspace(6.7,df['k'].max(),8000)

			int_func=interp1d(df['k'],df['mu'], kind='cubic')
			#int_funcz=np.polyfit(df['k'],df['mu'], deg=20)

			for j in interpx:
				smoothdf.append(int_func(j))

			zeros=[]

			for pp in range(len(interpx)-1):
				if smoothdf[pp]*smoothdf[pp+1]<0:
					#print('eriuhnrejcn')
					zeros.append(interpx[pp])


			#print(kz_cross)
			kz_cross.append(zeros)

			plt.plot(interpx,smoothdf,label= str(datain).split('.')[0].split('_')[1].split('ff')[0],marker='o')
			#plt.plot(outt['k'],outt['mu'],label=str(datain))
			plt.legend(fontsize=16)
			plt.title('Back FF Br-Pb k-space')
			plt.xlabel('k ($\AA^{-1}$)')
			plt.ylabel('k*$\chi$ (k)')

			plt.show()




		return df, kz_cross, int_func, smoothdf



go=KData(datals)
outt=go.reader()
smoo=outt[3]
allzs=pd.DataFrame(outt[1])


plt.figure()
plt.title('Absolute Change in Zero-crossings of Back FFT Br data' )
#allzs.drop([0],axis=0,inplace=True)
c3=[]


def lin(x,a,b):
	return a*x+b
def cub(x,a,b):
	return a*x**3+b
def cub2(x,a,c,d):
	return a*x**3+c*x+d
def c3xtract(c_eff):
	return .75*np.pi*c_eff
def roxtract(r_eff):
	return  .5*np.pi*r_eff

for i in range(len(datals)):
	if i <7:
		plt.plot(temps,allzs[i]-allzs[i][0],label=str(i)+'th cross @ '+ str(allzs[i][1].round(2)) + ' $\AA^{-1}$',marker='o')
		plt.legend()
		plt.xlabel('Temp (K)')
		plt.ylabel('$\Delta k_0$ from lowT zero-crossing ($\AA^{-1}$)')
		popt, pcov = curve_fit(cub, temps,abs(allzs[i]-allzs[i][0]))
		c3.append(popt[0])
		c3ls=[]
#		for kk in temps:
#			c3ls.append(cub(popt[0],kk)/100.)
#		plt.plot(temps,c3ls)
allzsb=allzs.drop([10,11],axis=1)
rpb=[2.993599,2.994199,2.9947,2.9957,2.9988,3.0016,3.002,3.0074,3.0067,3.0058999,3.0116]
rbrfix=[2.9816,2.9832,2.9843,2.9884,2.9905,2.993,2.993,2.998,2.997,2.9991,3.0017]
rbrfree=[2.9833,2.985,2.98715,2.9809,2.980067,2.984417,2.97768,2.9672,2.96803,2.969,2.9639]
r0=2.993599
k0=allzsb.iloc[0]
c3_all_nor=pd.DataFrame().reindex_like(allzsb)
c3_all_br_r=pd.DataFrame().reindex_like(allzsb)
c3_all=pd.DataFrame().reindex_like(allzsb)
c3_cam=pd.DataFrame().reindex_like(allzsb)
for jj in range(len(allzsb.iloc[0])-3):
	for hh in range(len(allzsb[0])):
		print(allzsb[jj][hh]-allzsb[jj][0])
		print('funbffubfbuf')
		print(allzsb[jj][0]**3)

		# delta k = allzsb[jj][hh]-allzsb[jj][0]
		# delta r = rbrfree[hh]-rbrfree[0]

		c3_cam[jj][hh]=(3/2.)*(1/((allzsb[jj][0])-3*(allzsb[jj][hh]-allzsb[jj][0])))*((rbrfix[hh]-rbrfix[0])/(allzsb[jj][0])+(rbrfix[0])*(allzsb[jj][hh]-allzsb[jj][0])/(allzsb[jj][0]**2))
		c3_all_nor[jj][hh]=(3/2.*(allzsb[jj][hh]-allzsb[jj][0])*r0/(allzsb[jj][0]**3))*(1.+(3.)*(allzsb[jj][hh]-allzsb[jj][0])/allzsb[jj][0])**(-1)
		c3_all_br_r[jj][hh]=(3/2.*(allzsb[jj][hh]-allzsb[jj][0])*r0/(allzsb[jj][0]**3)+(rbrfree[hh]-rbrfree[0])/allzsb[jj][0]**2)*(1.+(3.)*(allzsb[jj][hh]-allzsb[jj][0])/allzsb[jj][0])**(-1)
		c3_all[jj][hh]=(3/2.*(allzsb[jj][hh]-allzsb[jj][0])*r0/(allzsb[jj][0]**3)+(rbrfix[hh]-rbrfix[0])/allzsb[jj][0]**2)*(1.+(3.)*(allzsb[jj][hh]-allzsb[jj][0])/allzsb[jj][0])**(-1)

'''
plt.figure()
for jk in range(len(allzsb[0])):
	plt.plot(allzsb.iloc[jk],c3_all.ix[jk],label=str(temps[jk])+'K',marker='o')
	plt.xlabel('k$_0$ crossing ($\AA^{-1}$)')
	plt.ylabel('Extracted C$_3$')
	plt.title('Extracted C$_3$ vs. K$_0$ for each Temp. (Br K) pb r.iloc ')
	plt.show()
	plt.legend()


plt.figure()
for jk in range(len(allzsb[0])):
	plt.plot(allzsb.iloc[jk],c3_all_br_r.ix[jk],label=str(temps[jk])+'K',marker='x')
	plt.xlabel('k$_0$ crossing ($\AA^{-1}$)')
	plt.ylabel('Extracted C$_3$')
	plt.title('Extracted C$_3$ vs. K$_0$ for each Temp. (Br K) br free r ')
	plt.show()
	plt.legend()

plt.figure()
for jk in range(len(allzsb[0])):

	plt.plot(allzsb.iloc[jk],c3_all_nor.ix[jk],label=str(temps[jk])+'K',marker='+')
	plt.xlabel('k$_0$ crossing ($\AA^{-1}$)')
	plt.ylabel('Extracted C$_3$')
	plt.title('Extracted C$_3$ vs. K$_0$ for each Temp. (Br K) no r correction ')
	plt.show()
	plt.legend()
	'''

c3_o_br=c3_all.mean(axis=1)
c3_o_br2=c3_cam.mean(axis=1)
c3_obars_br=c3_all.std(axis=1)
c3_obars_br2=c3_all.std(axis=1)
plt.figure()
#plt.errorbar(temps, c3_o_pb, yerr=c3_obars_pb, marker='o', label='pb')
plt.errorbar(temps, c3_o_br, yerr=c3_obars_br, marker='o', label='br')
plt.errorbar(temps, c3_o_br2, yerr=c3_obars_br2, marker='+', label='camrbr')

plt.legend()


c3_tot=pd.concat([pd.Series(temps), c3_o_br, c3_obars_br], axis=1)
c3_tot.columns=['temp','br','br_err']
#c3_tot.to_csv('c3_kcross_br.dat', sep='\t')

'''
# #################
#My attempt at fitting N(k) to extract c3 from the coeficient of the cubic term

plt.figure()
plt.title('N v K$_0$ for Each Temperature (Br-Pb Back FF)' )
allzsb=allzs.drop([10,11,12,13],axis=1)
c3b=[]
c3c=[]
for i in range(len(datals)):
	if i <10:
		plt.plot(allzsb.iloc[i],pd.Series(range(len(allzsb.ix[i])))+13,label=str(temps[i]),marker='o')
		plt.legend()
		plt.xlabel('# crossing')
		plt.ylabel('Zero-crossing ($\AA^{-1}$)')
		popt, pcov = curve_fit(cub2,  np.array(allzsb.iloc[i]), pd.Series(range(len(allzsb.ix[i])))+13)
		c3b.append(popt[0])
		c3c.append(popt[1])
		c3ls=[]
		for kk in np.linspace(4,14,100):
			c3ls.append(cub2(kk,popt[0],popt[1],popt[2]))
		plt.plot(np.linspace(4,14,100),c3ls)


plt.show()
df_ceff=pd.Series(c3b)
df_reff=pd.Series(c3c)

df_c=df_ceff.apply(c3xtract)
df_r=df_reff.apply(roxtract)
'''
