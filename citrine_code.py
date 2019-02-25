import ast
import pandas as pd
import copy
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import math
import os
import re
import copy
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold,train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor
from sklearn.svm import SVC,LinearSVR, SVR,LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, LinearRegression

np.seterr(divide='ignore')

#Just a way to reorder the dataset so that the the A/B values have some inherent meaning.
#One could just stick to max/min/ and binary operations instead.
def biggest_first(df,test=False):
	df_=copy.deepcopy(df)
	reversed=[]
	for i,row in df.iterrows():
		#Ranking by atomic volume, with the largest going in the 'A' slot.
		#If we are out of order, everything pair of columns have to switch around.
		if row.iloc[3]>row.iloc[2]:
			row_=row.values
			#The switch.
			for j in range(row_.shape[0]/2):
				row_[[2*j,2*j+1]]=row_[[2*j+1,2*j]]
			#The trick neede to reverse the stability vector string
			if not test:
				row_[-1] = str(ast.literal_eval(row_[-1])[::-1])
				df_.loc[i,:]=row_
			else:
				reversed.append(i)
	if test:
		return df_,reversed
	return df_

#A helper method for PCA derived dimensionality reduction.
#Want enough components to be able to explain pval of the explained variance.
def reduce_dim(desc,pval=0.8,return_pca=False):
	#The skip PCA case.
	if pval>=1.0:
		return desc
	pca =PCA(n_components=min(desc.shape))
	pca.fit(desc)
	#Find the needed number of components
	explained = np.cumsum(pca.explained_variance_ratio_)
	need_comps = np.argmax(explained>pval)+1
	#Just in case something goes wrong
	if need_comps ==1:
		need_comps = min(desc.shape)
	pca = PCA(n_components=need_comps)
	red_data = pca.fit_transform(desc)
	#Used if you need to save the pca transformation for general use.
	if return_pca:
		return pca,red_data
	return red_data	

#A large testing method for testing out a variety of classifiers.
#Can be used with no cross val. or 5 fold cross validation.
#pval determines the amount of dimensionality reduction.
#Can return error based on probability estimates and
#a precision/recall/f1-score classification dictionary.	
def first_classification_test(desc,stabs,pval=0.9,classifiers=[],cv=False):
	red_data = reduce_dim(desc,pval)
	print 'Dimensionality Reduced'
	#Create a default general set of classifiers.
	if len(classifiers)==0:
		for c in [0.01,0.025,0.1,0.5,1.]:
			#classifiers.append(LinearSVC(C=c))
			for pen in ['l1','l2']:
				classifiers.append(LogisticRegression(C=10*c,penalty=pen))
			for gamma in [0.5,1.,1.5,2.,5.]:
				classifiers.append(SVC(gamma=gamma,C=10*c,probability=True))
				if c==0.01:
					classifiers.append(GaussianProcessClassifier(RBF(gamma)))
		for depth in [5,10,15,25,100]:
			classifiers.append(DecisionTreeClassifier(max_depth=depth))
			for n_est in [10,15,25]:
				classifiers.append(RandomForestClassifier(max_depth=depth, n_estimators=n_est))
	acc_scores=[]
	class_dicts=[]
	print 'Starting Fits'
	for i, clf in enumerate(classifiers):
		#Run the appropriate helper methods depending on the value of cv.
		if cv:
			acc_score,class_dict = cv_classify_data(red_data,stabs,clf,n_splits=5)
			acc_scores.append(acc_score)
			class_dicts.append(class_dict)
		else:
			#Some vestigial error checking from a previous project.
			try:
				acc_score,class_dict = classify_data(red_data,stabs,clf)
				acc_scores.append(acc_score)
				class_dicts.append(class_dict)
			except:
				pass
	print 'Fits done'
	#Get a sort of 'base' accuracy score based on an intercept fit.
	if cv:
		bscore = cv_regress_data(np.ones((stabs.shape[0],1)),stabs,LinearRegression(),n_splits=5)
	else:
		bscore = regress_data(np.ones((stabs.shape[0],1)),stabs,LinearRegression())
	return bscore, acc_scores,class_dicts,classifiers	

#Helper method for basic classification	
def classify_data(descriptors,targets,classifier):
	classifier.fit(descriptors,targets)
	#Get classification information
	class_dict = classification_report(targets,classifier.predict(descriptors),output_dict=True)
	probs = classifier.predict_proba(descriptors)
	vals = []
	for p in probs:
		vals.append(p[0]*classifier.classes_[0]+p[1]*classifier.classes_[1]+p[2]*classifier.classes_[2])
	#Collect an RMSE based on the probabilities. Not the world's best metric/
	err = np.sqrt(np.mean((vals-targets)**2))
	return err,class_dict
	
#Helper method for cross-validated classification	
def cv_classify_data(descriptors,targets,classifier,n_splits=5):
	#Get a set of shuffled splits
	kf = KFold(n_splits=n_splits,shuffle=True)
	cv_errs=[]
	cv_class_dicts=[]
	preds = np.zeros(targets.shape)
	vals = np.zeros(targets.shape[0])
	#Run the n-fold cross validation
	for train_index,test_index in kf.split(descriptors):
		#Collect the train and validation sets.
		x_train = descriptors[train_index]
		x_test = descriptors[test_index]
		y_train = targets[train_index]
		y_test = targets[test_index]
		classifier.fit(x_train,y_train)
		#Collect the validation predictions
		for i in test_index:
			preds[i]=classifier.predict(descriptors[[i]])
			prob = classifier.predict_proba(descriptors[[i]])[0]
			val=prob[0]*classifier.classes_[0]+prob[1]*classifier.classes_[1]#+prob[2]*classifier.classes_[2]
			vals[i]=val
	#Had a mistake here. So saved data will be questionable. 
	class_dict = classification_report(targets,preds,output_dict=True)
	err = np.sqrt(np.mean((vals-targets)**2))
	return err,class_dict
	
#Take a regressor and run an n-fold CV and return the mean RMSE.	
def cv_regress_data(descriptors,targets,regressor,n_splits=5):
	kf = KFold(n_splits=n_splits,shuffle=True)
	errs=[]
	for train_index,test_index in kf.split(descriptors):
		x_train = descriptors[train_index]
		x_test = descriptors[test_index]
		y_train = targets[train_index]
		y_test = targets[test_index]
		regressor.fit(x_train,y_train)
		err = np.sqrt(np.mean((regressor.predict(x_test)-y_test)**2))
		errs.append(err)
	return np.mean(errs)

#Get better valence electron data to use as a descriptor.
#Somewhat deals with the mess of favorable halfshells and different subshells
#i.e. 4s1 3d4 may give up or want 1 s electron, 1 or 6 d electrons, etc.	
def get_valence_stuff(desc):
	#Lists for saving stuff. pref_ stuff is for what I think should be the best answer.
	#Correlations show I am wrong.
	pref_val_list_a=[]
	all_val_list_a=[]
	pref_uf_list_a=[]
	all_uf_list_a=[]
	all_val_list_b=[]
	pref_val_list_b=[]
	all_uf_list_b=[]
	pref_uf_list_b=[]
	for row in desc:
		#So I can do A and B in one run
		for j in range(2):
			ufs_=[]
			vals_=[]
			#Check for s/p block by seeing if d and f are full.
			if row[66+j] ==0 and row[70+j]==0:
				#Add p unfilled/valence and s+p uf/val to the lists
				vals_=[row[76+j],row[76+j]+row[80+j]]
				ufs_=[row[78+j],row[78+j]+row[74+j]]
				#If p block is empty
				if row[76+j]==0:
					prv =row[80+j]
				else:
					prv = row[76+j]
				#empty s stuff
				pru = row[78+j]
				#If p has something in it...
				if row[76+j]>0:
					pru+=row[74+j]
			#Handle d-block and f-block guys now
			else:
				#unfilled stuff first....
				su = row[78+j]
				pu = row[74+j]
				du = row[66+j]
				fu = row[70+j]
				#Half-filled stuff
				du2 = max(0,5-du)
				fu2 = max(0,7-fu)
				#Compute all favorable possible electron gains
				ufs_=[su,pu,du,fu,du2,fu2,su+du,su+fu,su+du+fu,fu+du,su+du2,su+fu2,su+du2+fu,su+du2+fu2,su+du+fu2,du2+fu,du+fu2,fu2+du2]
				ufs_.sort()
				#Remove any zero entries and repeat entries
				ufs_=list(set(ufs_))
				if ufs_[0]==0:
					ufs_.pop(0)
				#Preferred unfilled is filling up everything
				pru = su+pu+du+fu
				#now the valence stuff
				sv = row[80+j]
				pv = row[76+j]
				dv = row[68+j]
				fv = row[72+j]
				#Same half-filled and possible combinations as before
				dv2 = max(0,dv-5)
				fv2 = max(0,fv-7)
				vals_=[sv,pv,dv,fv,dv2,fv2,sv+dv,sv+fv,sv+dv+fv,fv+dv,sv+dv2,sv+fv2,sv+dv2+fv,sv+dv2+fv2,sv+dv+fv2,dv2+fv,dv+fv2,fv2+dv2]
				vals_.sort()
				vals_=list(set(vals_))
				if vals_[0]==0:
					vals_.pop(0)
				#Now handle by preferred valence/available electron guess
				prv=0
				#If a shell is unfilled, it's electrons are up for grabs.
				if du>0:
					prv+=dv
				if su>0:
					prv+=sv
				if pu>0:
					prv+=pv
				if fu>0:
					prv+=fv
			#Append the values to the appropriate lists.
			if j==0:
				all_val_list_a.append(vals_)
				all_uf_list_a.append(ufs_)
				pref_uf_list_a.append(pru)
				pref_val_list_a.append(prv)
			else:
				all_val_list_b.append(vals_)
				all_uf_list_b.append(ufs_)
				pref_uf_list_b.append(pru)
				pref_val_list_b.append(prv)

	return all_val_list_a,all_uf_list_a,all_val_list_b,all_uf_list_b,pref_val_list_a,pref_uf_list_a,pref_val_list_b,pref_uf_list_b
				
#Create a descriptor, from the valence electron stuff above, to create a descriptor
#that tries to describe the `electron compatibility' of a particular alloy.
def get_fit_desc(frac_a,frac_b,all_val_list_a,all_uf_list_a,all_val_list_b,all_uf_list_b,pref_val_list_a,pref_uf_list_a,pref_val_list_b,pref_uf_list_b):
	#Create a preferred and a first possible fit descriptor set
	fit_desc=[]
	pref_fit_desc=[]
	pref_val_sim=[]
	#For each alloy candidate, calculate all possible electron/hole matchups,
	#and pick the best fit (smallest difference).
	for i in range(len(all_val_list_a)):
		diffs=[]
		for v_a in all_val_list_a[i]:
			for u_b in all_uf_list_b[i]:
				diffs.append(np.abs(frac_a*v_a -frac_b*u_b))
		for u_a in all_uf_list_a[i]:
			for v_b in all_val_list_b[i]:
				diffs.append(np.abs(frac_a*u_a-frac_b*v_b))
		fit_desc.append(min(diffs))
		#Also grab the preferred differences.
		pref_fit_desc.append(min(np.abs(frac_a*pref_val_list_a[i]-frac_b*pref_uf_list_b[i]),np.abs(frac_b*pref_val_list_b[i]-frac_a*pref_uf_list_a[i])))
		pref_val_sim.append(np.abs(pref_val_list_a[i]-pref_val_list_b[i]))
	return fit_desc,pref_fit_desc,pref_val_sim
	
#Grab some perspective good/meaningful descriptors
#Based on correlation test from outside of this script.	
def get_additional_desc(frac_a,frac_b,desc):
	#Want, for now, atomic volume, boiling emp., melting temp., ICSD vol., miracle radius, elec_surf_dens., and gs_nrg
	stuff = np.zeros((desc.shape[0],12))
	#Absolute difference, signed difference, one addition, and ratio versions of the descriptors.
	adinds=[0,42,60]
	dinds = [4,56,14]
	for i,di in enumerate(adinds):
		stuff[:,i] = np.abs(desc[:,di]*frac_a - desc[:,di+1]*frac_b)
	for k,di in enumerate(dinds):
		stuff[:,i+k] = desc[:,di]*frac_a - desc[:,di+1]*frac_b
	i+=k
	stuff[:,i+1] = desc[:,22]*frac_a+desc[:,23]*frac_b
	i+=2
	rinds=[0,42,60,4,56]
	for j, ri in enumerate(rinds):
		stuff[:,i+j] = 1-((frac_a*desc[:,ri])/(frac_b*desc[:,ri+1]))
		'''if np.argwhere(np.isinf(stuff[:,i+j])).shape[0]>0:
			print i+j
			i1 = np.argwhere(np.isinf(stuff[:,i+j]))[0]
			print desc[i1[0],ri], desc[i1[0],ri+1]'''
	stuff[:,-5] = np.abs(stuff[:,-5])
	stuff[:,-4] = np.abs(stuff[:,-4])
	return stuff

#Create every version of the descriptor matrix.	
def get_full_desc_mat(desc,frac=0.5):
	desc_diff = np.zeros((desc.shape[0],desc.shape[1]/2))
	desc_adiff = np.zeros(desc_diff.shape)
	desc_add = np.zeros(desc_diff.shape)
	desc_prod = np.zeros(desc_diff.shape)
	desc_rat = np.zeros(desc_diff.shape)
	for i in range(desc_diff.shape[1]):
		desc_diff[:,i] = frac*desc[:,2*i]-(1-frac)*desc[:,2*i+1]
		desc_adiff[:,i]=frac*np.abs(desc[:,2*i]-(1-frac)*desc[:,2*i+1])
		desc_add[:,i] = frac*desc[:,2*i]+(1-frac)*desc[:,2*i+1]
		desc_prod[:,i] = frac*desc[:,2*i]*(1-frac)*desc[:,2*i+1]
		t = desc[:,2*i+1]
		for j,el in enumerate(t):
			if el==0:
				t[j] = 0.001
		desc_rat[:,i] = frac*desc[:,2*i]/((1-frac)*t)
	desc_mat = np.hstack((desc,desc_diff,desc_adiff,desc_add,desc_prod,desc_rat))
	#Standardize the result
	desc_mat = (desc_mat-np.mean(desc_mat,axis=0))/np.std(desc_mat,axis=0)
	return desc_mat
	
#Make an `ideal'/meaningful descriptor matrix for the compound yes/no problem.
def make_bool_desc_matrices(desc,stabs):
	#A Boolean matrix of whether or not any stable alloy is made between el.A and el.B
	comp_bin = [int(np.any(stabs[i])) for i in range(stabs.shape[0])]
	desc_mat = get_full_desc_mat(desc)
	vstuff = get_valence_stuff(desc)
	fds_ = get_fit_desc(0.5,0.5,*vstuff)
	fds_ = np.array(fds_).T
	#desc_mat = np.hstack((desc_mat,fds_[:,[0,2]]))
	#Get descriptor/descriptor correlations
	corr_mat = np.corrcoef(desc_mat.T)
	#Get all descriptor/target correlations
	corr_list = [np.corrcoef(desc_mat[:,j],y=np.array(comp_bin))[0][1] for j in range(desc_mat.shape[1])]
	good_inds_=range(desc_mat.shape[1])
	bads=[]
	#Remove anything that has a NaN or inf value.
	for gi in good_inds_:
		#print gi
		if np.isnan(desc_mat[:,gi]).any() or np.isinf(np.abs(desc_mat[:,gi])).any():
			bads.append(gi)
			#print gi
			#continue
	for gi in good_inds_:
		if gi in bads:
			continue
		#Remove descriptors with a low descriptor/target correlation
		if np.abs(corr_list)[gi]>=0.12:# and not np.isnan(desc_mat[:,gi]).any():
			for gi2 in copy.copy(good_inds_):
				if gi2 in bads:
					continue
				#If gi and gi2 have a high correlation, remove the one with the highest interdescriptor correlation
				if gi!=gi2:
					if np.abs(corr_mat[gi][gi2])>=0.70:
						if np.abs(corr_list)[gi2]>=0.12:
							s1 = np.sum(np.abs(corr_mat)[gi])
							s2 = np.sum(np.abs(corr_mat)[gi2])
							if s1>=s2:
								bads.append(gi)
								break
							else:
								bads.append(gi2)
								
						else:
							bads.append(gi2)
						
		else:
			bads.append(gi)
	good_inds = [gi for gi in good_inds_ if gi not in bads]
	return desc_mat[:,good_inds],good_inds
			
#Make statistically meaningful descriptor matrices for the alloy stabilities fits.			
def make_all_desc_matrices(desc,stabs,alloy_inds,target_corr=0.06,inter_corr=0.80,overlap_num=6):
	fracs_a = np.array([0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9])
	#Need to clean some of this up
	fracs_b = 1-fracs_a
	#nwdesc = get_additional_desc(0.5,0.5,desc)
	nwdesc=get_full_desc_mat(desc)
	dmats=[]
	for frac in fracs_a:
		#Replace this with a modified version of get_full_desc_mat with frac_a added in.
		#dmats.append(get_additional_desc(frac,1-frac,desc))
		dmats.append(get_full_desc_mat(desc,frac))
	#Handle the 50/50 case differently
	dmats.insert(4,nwdesc)
	#Remove non stable candidates from the boolean fit.
	for i,dmat in enumerate(dmats):
		dmats[i] = dmat[alloy_inds]
	#Append the electron descriptor to the matrix
	vstuff=get_valence_stuff(desc)
	fracs_a = list(fracs_a)
	#So GROSS
	fracs_a.insert(4,0.5)
	for i,frac in enumerate(fracs_a):
		s_,p,sim_=get_fit_desc(frac,1-frac,*vstuff)
		s_=np.array(s_).reshape(-1,1)
		sim_=np.array(sim_).reshape(-1,1)
		dmats[i]=np.hstack((dmats[i],s_[alloy_inds],sim_[alloy_inds]))
	#Standaradize each matrix
	for i,dmat in enumerate(dmats):
		dmats[i] = (dmat-np.mean(dmat,axis=0))/np.std(dmat,axis=0)
	#We have a weighted and unweighted matrix of potential descriptors 
	#(unweighted = 50/50 case)
	max_ind = dmats[0].shape[1]*2-2
	good_inds = [range(max_ind)]*9
	alloys= stabs[alloy_inds]
	
	#Search through each alloy matrix, looking for meaningful and independent descriptors.
	for i,dmat in enumerate(dmats):
		bads =[]
		#Remove columns with NaN and inf values
		for j in range(dmat.shape[1]):
			if np.isnan(dmat[:,j]).any() or np.isinf(np.abs(dmat[:,j])).any():
				bads.append(j)
			if np.isnan(dmats[4][:,j]).any() or np.isinf(np.abs(dmat)[:,j]).any():
				bads.append(j+dmat.shape[1]-1)
		#Non 50/50 case.
		if i!=4:
			#New matrix is weighted + unweighted + valence stuff.
			dm_ = np.hstack((dmat[:,:-2],dmats[4][:,:-2],dmat[:,[-2,-1]]))
			dmats[i]=dm_
			#Get descriptor/descriptor and descriptor/target correlations
			corr_mat = np.corrcoef(dm_.T)
			corr_list = [np.corrcoef(dm_[:,j],y=alloys[:,i])[0][1] for j in range(max_ind)]
			good_inds_=copy.copy(good_inds[i])
			for gi in good_inds_:
				if gi in bads:
					continue
				#Remove elements if they contain bad values or a a low target correlation value.
				if np.abs(corr_list)[gi]>=target_corr and not np.isnan(dm_[:,gi]).any() and not np.isinf(np.abs(dm_[:,gi])).any():
					for gi2 in copy.copy(good_inds_):
						if gi!=gi2:
							if gi2 in bads:
								continue
							#Pick the least correlated value of a highly correlated pair.
							if np.abs(corr_mat[gi][gi2])>=inter_corr:
								if np.abs(corr_list)[gi2]>=target_corr:
									s1 = np.sum(np.abs(corr_mat)[gi])
									s2 = np.sum(np.abs(corr_mat)[gi2])
									if s1>=s2:
										bads.append(gi)
										break
									else:
										bads.append(gi2)
										
								else:
									bads.append(gi2)
								
				else:
					bads.append(gi)
			good_inds[i]=[gi for gi in good_inds_ if gi not in bads]
		#Handle the 50/50 case
		else:
			corr_mat = np.corrcoef(dmat.T)
			corr_list = [np.corrcoef(dmat[:,j],y=alloys[:,i])[0][1] for j in range(dmat.shape[1])]
			good_inds_ = range(dmat.shape[1])
			for gi in good_inds_:
				if gi in bads:
					continue
				if np.abs(corr_list)[gi]>=target_corr:
					for gi2 in copy.copy(good_inds_):
						if gi2 in bads:
							continue
						if gi!=gi2:
							if np.abs(corr_mat[gi][gi2])>=inter_corr:
								if np.abs(corr_list)[gi2]>=target_corr:
									s1 = np.sum(np.abs(corr_mat)[gi])
									s2 = np.sum(np.abs(corr_mat)[gi2])
									if s1>=s2:
										bads.append(gi)
										break
									else:
										bads.append(gi2)
										
								else:
									bads.append(gi2)
								
				else:
					bads.append(gi)
			good_inds__ = [gi for gi in good_inds_ if gi not in bads]
			#Add in indices that would be from the copied non-weighted matrix.
			for gi in copy.copy(good_inds__):
				if gi< dmat.shape[1]-1:
					good_inds__.append(gi+dmat.shape[1])
			good_inds[4]=good_inds__
	
	#Only use the selected descriptors if they are meaningful to a certain number of alloy problems
	master_good=[]
	for i in range(max_ind):
		ct=0
		for gis in good_inds:
			if i in gis:
				ct+=1
		if ct>=overlap_num:
			master_good.append(i)
	dmats[4] = np.hstack((dmats[4][:,:-1],dmats[4]))
	#print dmats[0].shape
	for i,dmat in enumerate(dmats):
		dmats[i]=dmat[:,master_good]
		
	return dmats,master_good,good_inds
	

#A method for running a bunch of classifier tests on all of the different alloy classification problems.	
def alloy_class_test(dmats,stabs,alloy_inds,classifiers=[]):
	if len(classifiers)==0:
		for c in [0.025,0.1,0.25,0.5]:
			classifiers.append(LogisticRegression(C=c,penalty='l1'))
		for l in [0.1,5,15,40]:
			classifiers.append(GaussianProcessClassifier(RBF(l)))
		for c in [5.0,10.0,15.0,20.0]:
			classifiers.append(SVC(C=c,probability=True,gamma=3.5))
		for d in [15,20,25]:
			for m in [2,3]:
				classifiers.append(DecisionTreeClassifier(max_depth=d,min_samples_split=m))
		for d in [20,25,35]:
			for n in [20,25]:
				for m in[2,2]:
					classifiers.append(RandomForestClassifier(max_depth=d,min_samples_split=m,n_estimators=n))
		for d in [50,65]:
			for n in [20,25]:
				classifiers.append(RandomForestClassifier(max_depth=d,min_samples_split=2,n_estimators=n,criterion='entropy'))
	res_dict={}
	#bstr = 'alloy_gen2_2'
	#Run the test for each 9 alloy composition with and without PCA dimensionality reduction
	for i,dmat in enumerate(dmats):
		b,accs,cdicts,clfs = first_classification_test(dmat,stabs[alloy_inds][:,i],pval=0.9,classifiers=classifiers,cv=True)
		f1scores=[cdicts[j]['weighted avg']['f1-score'] for j in range(len(clfs))]
		res_dict['alloy%i_gen2_2' %((i+1)*10)]={'accs':accs,'cdicts':cdicts,'clfs':copy.deepcopy(clfs),'f1scores':f1scores}
		b,accs,cdicts,clfs = first_classification_test(dmat,stabs[alloy_inds][:,i],pval=1,classifiers=classifiers,cv=True)
		f1scores=[cdicts[j]['weighted avg']['f1-score'] for j in range(len(clfs))]
		res_dict['alloy%i_gen2_2f' %((i+1)*10)]={'accs':accs,'cdicts':cdicts,'clfs':copy.deepcopy(clfs),'f1scores':f1scores}
	pickle.dump(res_dict,open('alloy_script_test3.pckl','wb'))
	return res_dict

#Utilize information about correlations between alloy types to change predictions
#e.g. if a stable 10/90 alloy means a 50/50 alloy is unlikely, use this information.
#Maybe redo with alloy_inds removed?	
def finalize_prediction(prediction_probs,stab_corr,mult=0.5):
	new_predict=[]
	#Rescale the 0-1 probabilities to a sigmoidal value of -1 - 1.
	prediction_probs=list(np.array(prediction_probs)*2 - 1.)
	#For each stability vector predictions, modify each entry appropriately.
	#Move down each of the 9 entries in vector, adjusting them independently of the rest.
	#Thus, this is a zeroth order deal.
	for i,p in enumerate(prediction_probs):
		mod =0
		for j in range(len(prediction_probs)):
			if i!=j:
				#Based on another project. Basically, effects the distance to 1 or -1.
				#If we have a pos. probability but some other correlation that say otherwise, this will result in a larger change.
				try:
					v_ = (1-(np.sign(stab_corr[i][j])*np.sign(prediction_probs[j])*p))*mult*stab_corr[i][j]*prediction_probs[j]
				#Some old error catching
				except:
					print type(prediction_probs)
					print type(p)
					raise
				#The change is cumulative across the other 8 entries.
				mod+=v_
		#Want the mean change, of course.
		new_predict.append(p+(mod/len(prediction_probs)))
	#Recast as a boolean. [0-1] = stable, [-1-0] = unstable.
	new_predict_ = [int(npred>=0) for npred in new_predict]
	return new_predict_
	
#A massively overlong method for taking in data and classifiers, and returning a predicted set of stability vectors.
##needs to be cleaned up a bit.
#The main idea is to be able to perform training fits/predictions, 
#but also able to take in pretrained `stuff' and return a test set of predictions.
#test being true means we need a prefit/determined classifiers, descriptor indices, pca reducers, and stability correlation matrix (for modify=True).
#If return_bs, return all of the objects from the training run for use in a future test run. 
def create_predictions(data,bool_clf=None,alloy_clfs=None,test=False,return_bs=False,bool_inds=None,bool_pca=None,alloy_mat_inds=None,alloy_pca=None,stab_corr=None,modify=True):
	#Make sure we can run the test set of data if test=True
	assert bool_clf is not None and alloy_clfs is not None
	if test:
		rev_=[]
		assert np.array([blah is not None for blah in (bool_inds,bool_pca,alloy_mat_inds,alloy_pca)]).all()
		assert type(alloy_clfs)==list
		if modify:
			assert stab_corr is not None

	#Get the descriptor and stability vector data.
	#Could be given as a file name, a desc matrix, or a desc/stab tuple or list.
	if type(data)==str:
		df = pd.read_csv(data)
		data = biggest_first(df,test)
		if test:
			rev_=data[1]
			desc = data[0].values[:,2:]
			desc = np.array(desc,dtype=float)
	if type(data)==list:
		desc=data[0]
		#Hacky AF. Can just be blank.....
		stabs=data[1]
	else:
		if not test:
			desc = data[:,:-1]
			stabs = np.array(data[:,-1],dtype=int)[:,1:-1]
		
	#########################################################
	#              Compound Boolean Problem                 #
	#########################################################
	
	#Now create the descriptor matrix for the Boolean problem, either from old indices or from scratch.
	if bool_inds is None:
		bool_mat,bool_inds=make_bool_desc_matrices(desc,stabs)
	else:
		desc_mat = get_full_desc_mat(desc)
		#print max(bool_inds),desc_mat.shape
		bool_mat = desc_mat[:,bool_inds]
	#Perform the appropriate dimensionlity reduction
	if bool_pca is None:
		bool_pca,bool_mat2 = reduce_dim(bool_mat,pval=0.9,return_pca=True)
	else:
		bool_mat2=bool_pca.transform(bool_mat)
	#Now predict whether or not any compound will form. If we are in the train case, fit the classifier first.
	if test:
		preds = bool_clf.predict(bool_mat2)
	else:
		#Now create the compound yes/no Boolean vector
		comp_bin = [int(np.any(stabs[i])) for i in range(stabs.shape[0])]
		bool_clf.fit(bool_mat2,comp_bin)
		preds = bool_clf.predict(bool_mat2)
	#Use the results to determine the `interesting' alloy candidates
	alloy_inds = np.nonzero(preds>0)[0]
	
	##############################################
	#        Alloy Stability Determination       #
	##############################################
	
	#create the set of 9 alloy descriptor matrices, either from old indices or from scratch.
	if alloy_mat_inds is None:
		alloy_mats,alloy_mat_inds,g=make_all_desc_matrices(desc,stabs,alloy_inds)
	else:
		#Make the 9 candidate descriptor matrices
		#alloy_mats =[get_additional_desc(i/10.,1-(i/10.),desc)[alloy_inds] for i in range(1,10)]
		alloy_mats = [get_full_desc_mat(desc,i/10.)[alloy_inds] for i in range(1,10)]
		#Add in the valence/hole descriptor
		vstuff=get_valence_stuff(desc)
		add_nw = alloy_mats[4]
		i=0
		#print np.hstack((alloy_mats[i],add_nw,np.array(get_fit_desc((i+1)/10.,1-(i+1)/10.,*vstuff)).T[:,[0,2]][alloy_inds])).shape
		for i in range(9):
			alloy_mats[i]=np.hstack((alloy_mats[i],add_nw,np.array(get_fit_desc((i+1)/10.,1-(i+1)/10.,*vstuff)).T[:,[0,2]][alloy_inds]))[:,alloy_mat_inds]
			alloy_mats[i]=(alloy_mats[i]-np.mean(alloy_mats[i],axis=0))/np.std(alloy_mats[i],axis=0)
	#Perform the apporpriate dimensionality reduction
	if alloy_pca is None:
		alloy_pca=[]
		for i in range(9):
			ap,am = reduce_dim(alloy_mats[i],return_pca=True)
			alloy_pca.append(ap)
			alloy_mats[i]=am
	else:
		#Had some issues here with nan and inf elements.
		try:
			alloy_mats=[alloy_pca[i].transform(alloy_mats[i]) for i in range(9)]
		except:
			#print alloy_mats[0].shape
			print i
			print alloy_mats[i].shape
			print alloy_mats[i][:,np.argwhere(np.isnan(alloy_mats[i]))[0][1]]
			raise
	#Now run the classifiers for the 9 alloy types, fitting if necessary		
	preds=[]
	#We are also collecting the probabilities for use if modify = True (#dicey)
	preds_prob=[]
	if type(alloy_clfs)==list:
		preds = [alloy_clfs[i].predict(alloy_mats[i]) for i in range(9)]	
		preds_prob = [alloy_clfs[i].predict_proba(alloy_mats[i]) for i in range(9)]
	else:
		alloy_clf = copy.deepcopy(alloy_clfs)
		alloy_clfs=[]
		for i in range(9):
			alloy_clf.fit(alloy_mats[i],stabs[alloy_inds][:,i])
			preds.append(alloy_clf.predict(alloy_mats[i]))
			preds_prob.append(alloy_clf.predict_proba(alloy_mats[i]))
			if return_bs:
				alloy_clfs.append(copy.deepcopy(alloy_clf))
    #########################################################
    #           Finalization and Return of Results          #
    #########################################################	
	
	#Return results for future test runs
	if return_bs:
		return bool_clf,alloy_clfs,bool_inds,alloy_mat_inds,bool_pca,alloy_pca
	#Modify the results using the stability vector correlations.
	if modify:
		if stab_corr is None:
			stab_corr = np.corrcoef(stabs.T)
		preds=[]
		preds_prob =list(np.array(preds_prob).T)
		for p in preds_prob[1]:
			#mult should be a method parameter
			preds.append(finalize_prediction(p,stab_corr,mult=.5))
	sz = desc.shape[0]
	preds_=[]
	#Make sure our result matrices have the shape (alloy_candidates, alloy_composition)
	if np.array(preds).shape[0]<np.array(preds).shape[1]:
		preds = list(np.array(preds).T)
	#Create the final predicted stability matrix. Re-insert predicted unstable alloy candidates from the Boolean problem.
	for i in range(sz):
		if i in alloy_inds:
			preds_.append(preds[list(alloy_inds).index(i)])
		else:
			preds_.append([0.,0.,0.,0.,0.,0.,0.,0.,0.])
	if test:
		return preds_, rev_
	return preds_	
	
	
#Run a 632 Bootstrap test for a given data/bool_clf/alloy_clf set.	
def bootstrap_test(data,bool_clf,alloy_clf,modify=True,runs=150):
	#I was overly afraid of some kind of initialization error. Fixed it upstream though. 
	bc = copy.deepcopy(bool_clf)
	#Grabbing the random sample, with replacement
	prng = np.random.RandomState(int(23))
	#Getting the base training error (analagous to RMSE and not the EPE)
	full_preds = create_predictions([data[0],data[1]],bc,alloy_clf,modify=modify)
	#Allows us to record the percent of incorrect predictions.
	#Can also recover the truth using preds-diff.
	full_diff = np.array(full_preds)-np.array(data[1])
	full_adiff = np.abs(full_diff)
	#Run the bootstrapping process
	diffs=[]
	adiffs=[]
	bs_preds=[]
	for i in range(runs):
		good=False
		#Just a status indicator
		if i%10==0:
			print i
		while not good:
			#Get the random bag of indices
			rand_inds = np.floor(data[0].shape[0]*prng.rand(data[0].shape[0])).astype(int)
			rand_desc = data[0][rand_inds]
			rand_stabs=data[1][rand_inds]
			stab_corr_=np.corrcoef(rand_stabs.T)
			#Get the unused indices as test indices
			test_inds =[j for j in range(data[0].shape[0]) if j not in rand_inds]
			test_desc = data[0][test_inds]
			test_stabs=data[1][test_inds]
			#Make sure we do not end up with a test target that has a column of all ones or zeros. Ruins the corrcoef call.
			if np.array([np.array(test_stabs)[:,i].all() for i in range(9)]).any() or not np.array([np.array(test_stabs)[:,i].any() for i in range(9)]).all():
				good=False
			else:
				good=True
		bc = copy.deepcopy(bool_clf)
		#Create the materials from the training run
		stuff = create_predictions([rand_desc,rand_stabs],bc,alloy_clf,modify=modify,return_bs=True)
		#Need a set of all preds? Multiclass precision/recall?
		#Run the psuedo test prediction set
		preds_ = create_predictions([test_desc,test_stabs],stuff[0],stuff[1],test=True,modify=modify,bool_inds=stuff[2],alloy_mat_inds=stuff[3],bool_pca=stuff[4],alloy_pca=stuff[5],stab_corr=np.corrcoef(rand_stabs.T))[0]
		diff_ = np.array(preds_)-np.array(test_stabs)
		adiff_ = np.abs(diff_)
		diffs.append(diff_)
		adiffs.append(adiff_)
		bs_preds.append(preds_)
	#TEMPORARY
	#return full_preds,full_diff,full_adiff,bs_preds,diffs,adiffs
	
	#Collect and create some error metrics.
	#Focus on accuracy (percent of incorrect predictions) and typical precision/recall/f1-score metrics
	#Stack all of the Bootstrap values into one very tall matrix
	full_bs_adiff = np.vstack(tuple(adiffs))
	full_bs_diff = np.vstack(tuple(diffs))
	full_bs_pred = np.vstack(tuple(bs_preds))
	full_bs_true = full_bs_pred-full_bs_diff
	#Take the mean of this and then add in the 632 correction
	bs_acc = np.mean(full_bs_adiff)
	bs632_acc = np.exp(-1)*np.mean(full_adiff)+(1-np.exp(-1))*bs_acc
	#Now create and collect all of the classification metrics, both raw and corrected bootstrap numbers
	bs_cdicts = [classification_report(full_bs_true[:,i],full_bs_pred[:,i],output_dict=True) for i in range(9)]
	full_cdicts = [classification_report(np.array(data[1])[:,i],np.array(full_preds)[:,i],output_dict=True) for i in range(9)]
	bs_precisions = [bs_cdicts[i]['weighted avg']['precision'] for i in range(9)]
	bs_recalls = [bs_cdicts[i]['weighted avg']['recall'] for i in range(9)]
	bs_f1scores = [bs_cdicts[i]['weighted avg']['f1-score'] for i in range(9)]
	bs632_precisions = list((1-np.exp(-1))*np.array(bs_precisions) + np.exp(-1)*np.array([full_cdicts[i]['weighted avg']['precision'] for i in range(9)]))
	bs632_recalls = list((1-np.exp(-1))*np.array(bs_recalls) + np.exp(-1)*np.array([full_cdicts[i]['weighted avg']['recall'] for i in range(9)]))
	bs632_f1scores = list((1-np.exp(-1))*np.array(bs_f1scores) + np.exp(-1)*np.array([full_cdicts[i]['weighted avg']['f1-score'] for i in range(9)]))
	
	return bs_acc,bs632_acc,full_diff,full_adiff,diffs,adiffs,bs632_precisions, bs632_recalls, bs632_f1scores, bs_precisions, bs_recalls, bs_f1scores, full_cdicts, bs_cdicts
	
#Could make into a class, but whatever for now
#This will turn a test csv into a test csv with a predicted stability vector!
def make_predictions(data = 'test_data.csv',output_file='fit_test_data.csv',classifier_file = 'final_class_dict.pckl'):
	clf_dict = pickle.load(open(classifier_file,'rb'))
	preds,reverse = create_predictions(data,test=True,**clf_dict)
	df = pd.read_csv(data)
	stabs=[]
	#To look better
	for i,pred in enumerate(preds):
		for j,p in enumerate(pred):
			preds[i][j] = float(p)
	#To unreverse the dataframe
	for i,p in enumerate(preds):
		if i in reverse:
			stabs.append([1.0]+p[::-1]+[1.0])
		else:
			stabs.append([1.0]+p+[1.0])
	df['stabilityVec']=stabs
	df.to_csv(output_file)
	