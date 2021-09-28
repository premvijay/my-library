"""Credits: Sujatha Ramakrishnan"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg



class parameter_estimation(object):

	#~ def __init__(self,Omega_matter = 0.3):
		#~ self.Omega_matter=Omega_matter

	def chisquare(self,data,model,std_dev_error):
		"""
		axis keyword is for the axis which bears different data points for a fixed model
		"""
		chisquare = np.sum((data-model)**2/(std_dev_error**2+1e-15))
		return chisquare


	def probability_contours(self,sigma_1,sigma_2,gamma,p ,no_of_points = 100):
		"""
		Inputs
		
		sigma_1 --> The standard deviation of parameter 1
		sigma_2 --> The standard deviation of paramente 2
		gamma = sigma_12/ (sigma_1 * sigma_2) --> dimensionless qty which gives back the covariance
		p --> is a list of all the probability contours required
		Note: These elements are the inverse of the fisher matrix
		"""
		no_of_points = int(no_of_points)
		sigma_12 = gamma * sigma_1 * sigma_2
		fisher_inverse = np.matrix([[sigma_1**2,sigma_12**2],[sigma_12**2,sigma_2**2]])
		print (fisher_inverse,np.linalg.eig(fisher_inverse))
		chisq = -2*np.log(1-p)
		eig , v = np.linalg.eigh(fisher_inverse)
		xmin = - np.sqrt(eig[0]*chisq)
		xmax =   np.sqrt(eig[0]*chisq)
		x = np.linspace(xmin,xmax,no_of_points)
		y1 = np.sqrt(eig[1]*(chisq - x**2/eig[0]))
		y2 = - np.sqrt(eig[1]*(chisq - x**2/eig[0]))
		X = np.concatenate([x,x[::-1]])
		Y = np.concatenate([y1,y2[::-1]])
		print (eig**(1/2.))
		Rot = np.matrix(v)
		val = int(round(np.linalg.det(Rot)))
		if (val ==-1):
			Rot[:,0]=-Rot[:,0]
		Pos_mat = np.matrix(np.array([X,Y]))
		Pos_mat_new = np.empty(Pos_mat.shape)
		Pos_mat_new = Rot*Pos_mat
		return np.array(Pos_mat_new)


	def custom_polynomialfit(self,x,y,degree,sigma):
		"""
		Inputs
		x : value of x data points
		y : value of y data points
		degree : degree of the polynomial fit
		sigma: error in the value of y
		Returns:
		d: coefficients for the fit -->d[0]*x+d[1]*x**2+d[2]*x**3+...d[degree-1]*x**degree
		C: covariance matrix
		"""
		x = np.array(x)
		y = np.array(y)
		sigma = np.array(sigma)
		
		b = np.zeros([degree])
		# ~ f = np.zeros([degree+1,degree+1])
		f = np.zeros([degree,degree])
		# ~ for m in range(degree+1):
		for m in range(1,degree+1):
			b[m-1] = np.sum(1/sigma**2 *y*x**m)
			for n in range(1,degree+1):
				f[m-1,n-1] = np.sum(1/sigma**2 * x**m * x**n ) 
		b = np.matrix(b)
		f = np.matrix(f)
		u, s, vh = np.linalg.svd(f)
		finv = np.transpose(vh) * np.diag(1/s) * np.transpose(u)
		d = finv*b.T
		return d , finv
		
	def polynomialfit(self,x,y,degree,sigma):
		"""
		Inputs
		x : value of x data points
		y : value of y data points
		degree : degree of the polynomial fit
		sigma: error in the value of y
		Returns:
		d: coefficients for the fit -->d[0]+d[1]*x+d[2]*x**2+...d[degree]*x**degree
		C: covariance matrix
		"""
		x = np.array(x)
		y = np.array(y)
		sigma = np.array(sigma)
		
		b = np.zeros([degree+1])
		f = np.zeros([degree+1,degree+1])
		# ~ f = np.zeros([degree,degree])
		for m in range(degree+1):
		# ~ for m in range(1,degree+1):
			b[m] = np.sum(1/sigma**2 *y*x**m)
			for n in range(degree+1):
				f[m,n] = np.sum(1/sigma**2 * x**m * x**n ) 
		b = np.matrix(b)
		f = np.matrix(f)
		d = f.I*b.T
		return np.array(d).flatten() , f.I

	def polyval(self,b,x):
		"""
		gives y_model for polynomial fit
		"""
		ymodel=0
		for i in range(len(b)):
			ymodel+=x**i*b[i]
		return ymodel
			
		
	def AICriterion_corrected(self,chisquareML,p,N):
		"""
		N --> denotes number of points in the candidate model class
		p --> denotes the number of parameters in the model
		chisquareML --> denotes the chisquare of maximum liklihood
		AIC_corrected = chisquareML + 2pN/(N-p-1)
		The AIC values are subsequently ranked by the smallness of their values; the model class with the smallest value is determined to be the 'best' model class.
		"""
		return chisquareML + 2*p*N/(N-p-1)
	
	def BIC(self,chisquareML,p,N):
		return chisquareML + p*np.log(N)
		
	def AICpolynomial_fitter(self,x,y,sigma,xalt):
		"""
		AIC polynomial fitter for regular polynomials
		"""
		

		degree = 0
		AIC = np.zeros([6])
		BIC = np.zeros([6])
		maxpercentagediff = np.zeros([6])
		chisqpdof = np.zeros([6])
		for inter in range(6):
			d,f = self.polynomialfit(x,y,degree,sigma)
			y_model = np.zeros([len(x)])
			for power in range(len(d)):
				y_model += d[power]*x**(power)
				print (power)
			plt.errorbar(xalt,y,sigma,linestyle='None',elinewidth=5)
			plt.plot(xalt[np.argsort(xalt)],y_model[np.argsort(xalt)])
			plt.xscale('log')
			plt.show()
			chisq = self.chisquare(y,y_model,sigma)
			print ("chisqperdof",chisq/(len(x)-degree-1))
			chisqpdof[inter] = chisq/(len(x)-degree-1)
			AIC[inter] = self.AICriterion_corrected(chisq,degree+1,len(x))
			BIC[inter] = self.BIC(chisq,degree+2,len(x))
			maxpercentagediff[inter] = np.abs(y_model/y-1).max()
			print ('BIC is ',str(BIC[inter]))
			print (str(degree)+'polynomial with AIC',str(AIC[inter]))
			degree+=1
			d0 = d
			f0 = f
			print ('bestfit are:',d)
		return d0,f0,AIC,BIC,maxpercentagediff,chisqpdof
		
	def multifit(self,x1,y1,err1,x2,y2,err2,degree):
		""" Highly specific quadratic polynomial fitter for two data sets."""
		# ~ Y = np.zeros(4,dtype=float)
		Y = np.zeros(degree+2,dtype=float)
		g = np.zeros([degree+2,len(x1)])
		f = np.zeros([degree+2,len(x1)])
		f[0,:] = 1
		g[1,:] = 1
		for i in range(2,degree+2):
			g[i,:] = x2**(i-1)
			f[i,:] = x1**(i-1)
		# Matrix
		# ~ F = np.zeros((4,4),dtype=float)
		F = np.zeros((degree+2,degree+2),dtype=float)
		for i in range(degree+2):
			Y[i] = np.sum((y1*f[i,:])/err1**2)+np.sum((y2*g[i,:])/err2**2)
		for i in range(degree+2):
			for n in range(degree+2):	
				F[i,n] = np.sum(f[i,:]*f[n,:]/err1**2) + np.sum(g[i,:]*g[n,:]/err2**2)
		Y = Y.T
		U,s,Vh = linalg.svd(F)
		Cov = np.dot(Vh.T,np.dot(np.diag(1.0/s),U.T))
		a_minVar = Cov.dot(Y)
		return np.squeeze(np.asarray(a_minVar)),np.asarray(Cov)


	def multifit_quadratic(self,x1,y1,err1,x2,y2,err2):
		""" Highly specific quadratic polynomial fitter for two data sets."""
		Y = np.zeros(4,dtype=float)
   
		# Matrix
		F = np.zeros((4,4),dtype=float)
		Y[0] = np.sum(y1/err1**2)
		Y[1] = np.sum(y2/err2**2)
		Y[2] = np.sum(y1*x1/err1**2) + np.sum(y2*x2/err2**2)
		Y[3] = np.sum(y1*x1**2/err1**2) + np.sum(y2*x2**2/err2**2)
		F[0,0] = np.sum(1.0/err1**2)
		F[1,1] = np.sum(1.0/err2**2)
		F[2,2] = np.sum(x1**2/err1**2) + np.sum(x2**2/err2**2)
		F[3,3] = np.sum(x1**4/err1**2) + np.sum(x2**4/err2**2)
		F[0,1] = 0.0
		F[0,2] = np.sum(x1/err1**2)
		F[0,3] = np.sum(x1**2/err1**2)
		F[1,2] = np.sum(x2/err2**2)
		F[1,3] = np.sum(x2**2/err2**2)
		F[2,3] = np.sum(x1**3/err1**2) + np.sum(x2**3/err2**2)
		for alpha in range(0,4):
			for beta in range(alpha,4):
				F[beta,alpha] = F[alpha,beta]
		Y = Y.T
		U,s,Vh = linalg.svd(F)
		Cov = np.dot(Vh.T,np.dot(np.diag(1.0/s),U.T))
		a_minVar = Cov.dot(Y)
		return np.squeeze(np.asarray(a_minVar)),np.asarray(Cov)

	def corr(self,err,i,j):
		return err[i,j]/np.sqrt(err[i,i]*err[j,j])

	def maketable(self,nameoffile,bestfit,covmat,nameoffit,flag=0,chisq=None,dof=None):
		"""
		flag = 0 for polyfit from numpy
		flag = 1 for customfit from this class
		"""
		fhandle = open( nameoffile+'.tex', 'w' )
		fhandle.write( '\\renewcommand{\\arraystretch}{1.5} \n \\begin{tabular}{lllllll}\n \\hline \n \\hline  \n' )
		k = np.size(bestfit)-1
		firstrow = '&'
		secondrow = 'value&'
		thirdrow = 'std dev&'
		dictn = {}
		for loopi in range(k+1):
			dictn.update({str(loopi)+'th row':'corr ${'+nameoffit+str(loopi)+'}$&'})
		powr = 0
		for loopi in range(k+1):
			firstrow+='${'+nameoffit+str(loopi)+'}$&'
			secondrow+=format(bestfit[(1-flag)*(k-loopi)+flag*loopi],'5.4f')+'&'
			thirdrow+=format(np.sqrt(covmat[(1-flag)*(k-loopi)+flag*loopi,(1-flag)*(k-loopi)+flag*loopi]),'5.4f')+'&'
			for loopj in range(k+1):
				if (loopj<=loopi):
					dictn[str(loopj)+'th row']+=format(self.corr(covmat,(1-flag)*(k-loopi)+flag*loopi,(1-flag)*(k-loopj)+flag*loopj),'5.4f')+'&'
				else:
					dictn[str(loopj)+'th row']+='-&'
		if chisq==None:
			fhandle.write(firstrow+" \\\ ")
			fhandle.write("\\hline \n")
			fhandle.write(secondrow+" \\\ ")
		else:
			fhandle.write(firstrow+"$\chi^2(\\rm "+str(dof)+"\ d.o.f)$&\\\ ")
			fhandle.write("\\hline \n")
			fhandle.write(secondrow+"{:.2f}".format(chisq)+"&\\\ ")


		fhandle.write("\\hline \n")
		fhandle.write(thirdrow+" \\\ ")
		fhandle.write("\\hline \n")
		for loopj in range(k):
			fhandle.write(dictn[str(loopj)+'th row']+" \\\ ")
			fhandle.write("\\hline \n")
		fhandle.write(" \n \\hline \\\ \n")
		fhandle.write('\end{tabular}')
		fhandle.close()
	
