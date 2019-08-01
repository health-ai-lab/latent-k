#generate data for each causal relationship found in type level

import copy
import math
import time
import pandas as pd
import glob
import csv
import os
import numpy as np
import pprint as pp
import scipy.stats as ss
import random
import sys
import pickle

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def load_pickle(paramsfn):
        with open(paramsfn,'r') as f:
                mydata=pickle.load(f)
        set_causal_model_dic(mydata['causal_model_dic'])
        set_timeseries_leng(mydata['timeseries_leng'])
        set_causal_window_dic(mydata['causal_window_dic'])
        set_causal_window_pro_dic(mydata['causal_window_pro_dic'])
        set_max_lay(1) #default value is 1
        set_causal_model_layer({1:1}) #default first relationship on layer 1

#dictionary storing causal relationships in our causal model.{causal relationship tag: causal rela}
#EX:{1:["A","B"]}

gcausal_model_dic = {}

def get_causal_model_dic():
    global gcausal_model_dic
    return gcausal_model_dic

def set_causal_model_dic(causal_model_dic):
    global gcausal_model_dic
    gcausal_model_dic = causal_model_dic
    return gcausal_model_dic




#dictionary storing the layer tag for each causal rela starts with 1
#{causal rela tag: layer}



gcausal_model_layer = {}

def get_causal_model_layer():
    global gcausal_model_layer
    return gcausal_model_layer

def set_causal_model_layer(causal_model_layer):
    global gcausal_model_layer
    gcausal_model_layer = causal_model_layer
    return gcausal_model_layer

#stores the maximum layer

gmax_lay = 0

def get_max_lay():
    global gcausal_max_layer
    return gmax_lay

def set_max_lay(max_lay):
    global gcausal_max_layer
    gmax_lay = max_lay
    return gmax_lay


#list storing variables and their occurrence probabilities:{variable name: occur_pro}
#EX:[{"A":0.15}]

gvar_occur_lis = []
def get_var_occur_lis():
    global gvar_occur_lis
    return gvar_occur_lis

def set_var_occur_lis(var_occur_lis):
    global gvar_occur_lis
    gvar_occur_lis = var_occur_lis
    return gvar_occur_lis

#dictionary storing time windows for each causal relationship:
#{causal relationship tag: [min_lag,window_start,window_end,max_lag]}: {1:[1,2,3,5]}.

gcausal_window_dic ={}

def get_causal_window_dic():
    global gcausal_window_dic
    return gcausal_window_dic

def set_causal_window_dic(causal_window_dic):
    global gcausal_window_dic
    gcausal_window_dic = causal_window_dic
    return gcausal_window_dic



#the length of time series


gtimeseries_leng = 0

def get_timeseries_leng():
    global gtimeseries_leng
    return gtimeseries_leng

def set_timeseries_leng(timeseries_leng):
    global gtimeseries_leng
    gtimeseries_leng = timeseries_leng
    return gtimeseries_leng


#dictionary storing the probability distribution for each causal relationship
#{causal rela tag: type of distribution]}
#1:linear, 2:binary, 3: increase(slope), 4:decrease(slope)


#testing latent variables

glatent_var = []

def get_latent_var():
    global glatent_var
    return glatent_var

def set_latent_var(raw_data,causal_model_dic):
    global glatent_var
    Vd=raw_data.columns
    Vk=[]
    for item in causal_model_dic:
	for item2 in causal_model_dic[item]:
		Vk.append(item2)
    Vk=np.unique(Vk)
    Vd=np.unique(Vd)
    latent_var=[]
    for item in Vk:
	if Vk not in Vd:
		latent_var.append(item)
    glatent_var = latent_var
    var_occur_lis=[]
    for item in Vd:
	var_occur_lis.append({item:0.0})
    set_var_occur_lis(var_occur_lis)
    return glatent_var

#dictionary storing the probability for each time window: 
#{causal realtionship tag: pro}
#{1:0.8}

gcausal_window_pro_dic = {}

def get_causal_window_pro_dic():
    global gcausal_window_pro_dic
    return gcausal_window_pro_dic

def set_causal_window_pro_dic(causal_window_pro_dic):
    global gcausal_window_pro_dic
    gcausal_window_pro_dic = causal_window_pro_dic
    return gcausal_window_pro_dic



#saves the series without latent variables, but with the noise variables
def save_no_latent(raw_data,latent_var):
    #variable list
    raw_var_name = list(raw_data)
    #variables list without latent variables but including the noise variables
    nolatent_var_name = []
    for it in raw_var_name:
        if it not in latent_var:
            nolatent_var_name.append(it)

    nolat_data = raw_data[nolatent_var_name]
    nolat_data.to_csv(nolatentfn,index=False)


#load the raw time series
def load_timeseries(timeseriesfn,nolatentfn,inferseriesfn):
    timeseries_leng=get_timeseries_leng()
    causal_model_dic=get_causal_model_dic()
    #read data
    raw_data = pd.read_csv(timeseriesfn)
    latent_var=set_latent_var(raw_data,causal_model_dic)
    #save the series without latent variables, but with the noise variables
    save_no_latent(raw_data,latent_var)
    #get the processed series before inferring
    raw_var_name = list(raw_data)
    infer_data = {}
    for q in raw_var_name:
	infer_data[q]=raw_data[q][:timeseries_leng]
    #set the series needs inferring to list filled with zeros
    for it in latent_var:
        infer_data[it]=(np.zeros(timeseries_leng,dtype=np.int)).tolist()
    infer_data=pd.DataFrame(infer_data)
    #get all the variables in our model (exclude the noise variable)
    temp_all_var = []
    for k,v in causal_model_dic.items():
        temp_all_var.append(v[0])
        temp_all_var.append(v[1])
    all_vari_lis = list(set(temp_all_var))
    infer_series(raw_data,all_vari_lis,infer_data,latent_var,inferseriesfn)


def infer_series(raw_data,all_vari_lis,infer_data,latent_var,inferseriesfn):
    causal_model_dic=get_causal_model_dic()
    causal_model_layer=get_causal_model_layer()
    max_lay=get_max_lay()
    causal_window_dic=get_causal_window_dic()
    timeseries_leng=get_timeseries_leng()
    causal_window_pro_dic=get_causal_window_pro_dic()
    var_occur_lis=get_var_occur_lis()
    whole_count = 0
    real_count = 0
    #start infering the series
    #dictionary storing the probability for each latent variable
    #{lat_varlabel:[{pk:[evidences]}]}, each evidece is a tuple: (label,time,value)
    #e.g.{"H_t":[{0.33:[("A_3",0),("B_4",1),...]},...],"H_t2":[...]}
    lat_obser_pk = {}
    #get the no noise data
    no_noise_data = raw_data[all_vari_lis]
    #inferring the series
    #for each latent variable A-->H-->B, H is latent the time label for H is t, for A is t-1, for B is t+1
    # t = 1) layer number +1 (H is an effect) 2) layer number (H is an cause for that relationship)
    # we compute the time label for other variables using the same strategy
    #e.g. A-->H-->B, A-->C-->D. We have ("H",2),("A",1),("B",3),("C",2), ("D",3)
    
    #using the first latent varialbe to label all the variables
    if len(latent_var)==0:
        print "Error: no latent variable"
        sys.exit(1)
    fvar = latent_var[0]
    #get the minum time label for the latent variable( due to
    # feedback loops, the latent variable can have multiple labels)
    min_lat_label = 0
    temp_lay_lis = []
    for it_k,it_v in causal_model_dic.items():
        #latent var is an effect, lay+1
        if it_v[0]==fvar:
            temp_lay_lis.append(causal_model_layer[it_k])
        if it_v[1]==fvar:
            temp_lay_lis.append(causal_model_layer[it_k]+1)
    min_lat_label = min(temp_lay_lis)

    #get all the relationships with their variable labels: based on the latent variable's label
    #e.g. A1-->B2, B2-->A3 for feedback loops.
    all_edg_label = []
    #labels for all variables
    all_varlabels= []
    temp_allvarlabel = []
    #e.g. [(A_1,B_1),...] 
    for rel_k,rel_v in causal_model_dic.items():
        #for other variables in our model (not including latent variables),its layer number is
        # = min_lat_label + (target relationship label - min_Lat_label) +1 (if the target variable is an effect of that realtionship)
        #  or = min_lat_label + (target relationship label - min_Lat_label) +0 (if the target variable is a cause of that realtionship) 
        #get the label for both cause and effect
        ca_lab = min_lat_label+(causal_model_layer[rel_k] - min_lat_label)
        ef_lab = min_lat_label+(causal_model_layer[rel_k] - min_lat_label) +1
        all_edg_label.append((rel_v[0]+"_"+str(ca_lab),rel_v[1]+"_"+str(ef_lab)))
        temp_allvarlabel.append(rel_v[0]+"_"+str(ca_lab))
        temp_allvarlabel.append(rel_v[1]+"_"+str(ef_lab))
    all_varlabels = list(set(temp_allvarlabel))
    #Build the bayesian model
    whole_bayes = BayesianModel()
    #add edges
    for eg_it in all_edg_label:
        whole_bayes.add_edge(eg_it[0],eg_it[1])
    #defining individual CPDs for all origin causes (have own causes themeselves)
    #origin causes with labels
    origin_caus = []
    for b_var in var_occur_lis:
        #check whether it is an origin cause and not a noise variable
        for v_key,v_val in b_var.items():
            if v_val!=0 and v_key in all_vari_lis:
                #get the minimum cause label
                tmp_la = max_lay+1
                for eg_it in all_edg_label:
                    caef_lis = eg_it[0].split("_")
                    if str(caef_lis[0])==v_key and int(caef_lis[1])<=tmp_la:
                        tmp_la = int(caef_lis[1])
                #addd the individual cpds
                temp_cpd = str("cpd_")+str(v_key)+"_"+str(tmp_la)
                #print v_key+"_"+str(tmp_la)
                temp_cpd = TabularCPD(variable=v_key+"_"+str(tmp_la),variable_card=2,values=[[1-v_val,v_val]])
                #print v_key+"_"+str(tmp_la)
                whole_bayes.add_cpds(temp_cpd)
                origin_caus.append(v_key+"_"+str(tmp_la))

    if len(all_varlabels)!= len(list(set(all_varlabels))) or len(all_edg_label)!=len(list(set(all_edg_label))):
        print "Error: all_varlabels or all_edge_label has duplicates"
        sys.exit(1)
    #defining the indivial CPDs for other variables 
    for v_key in all_varlabels:
        #exclude the origin causes (since we already excluded the noise variable before in all_varlabels)
        if v_key not in origin_caus:
            #get all the causes for this variable
            #storing the causal rela index for all relationshps tat has v_key as their effect
            re_ind = []
            #all causes
            temp_caus_vars = []
            #cards for cpd
            temp_cards = []
            #the probability for each relationshp (with labels)
            temp_rela_prob=[]
            for eg_it in all_edg_label:
                if eg_it[1]==v_key:
                    temp_caus_vars.append(eg_it[0])
                    temp_cards.append(2)
                    #find the causal relationships probability
                    #find the index first
                    tm_r_in = 0
                    for r_k,r_v in causal_model_dic.items():
                        if r_v[0] in eg_it[0] and r_v[1] in eg_it[1]:
                            tm_r_in = r_k
                    #store the index
                    re_ind.append(tm_r_in)
                    temp_rela_prob.append(causal_window_pro_dic[tm_r_in])
            #compute the conditional proabability table for this variable v_key with labels
            #compute alll possible combinations
            temp_all_comb = []
            fir_comb = []
            for it in temp_caus_vars:
                fir_comb.append(0)
            temp_all_comb.append(fir_comb)
            for it in range(1,len(temp_caus_vars)+1):
                #copy
                temp_it_lis = copy.copy(temp_all_comb)
                for iit in range(len(temp_it_lis)):
                    temp_iit_lis = copy.copy(temp_it_lis[iit])
                    #modify
                    temp_iit_lis[len(temp_caus_vars)-it]=1
                    #update the raw list
                    temp_all_comb.append(temp_iit_lis)
            #compute the conditional probability
            temp_p_ta = []
            for ite in temp_all_comb:
                temp_pr_lis = []
                if sum(ite)==0:
                    temp_pr_lis.append(1.0)
                    temp_pr_lis.append(0.0)
                else:
                    #compute te case when the effect has 0 value
                    zero_pro = 1.0
                    for z_it in range(0,len(ite)):
                        if ite[z_it]==1:
                            #get the causal relationship's probability
                            zero_pro = zero_pro*(1.0-causal_window_pro_dic[re_ind[z_it]])
                    temp_pr_lis.append(zero_pro)
                    temp_pr_lis.append(1.0-zero_pro)
                #update the whole prob
                temp_p_ta.append(temp_pr_lis)
            #transpose
            all_temp_prob_ta = (np.asmatrix(temp_p_ta)).transpose()
            #add the CPDs
            temp_cpds_v = TabularCPD(variable=v_key,variable_card=2,values=all_temp_prob_ta,evidence=temp_caus_vars,evidence_card=temp_cards)
            whole_bayes.add_cpds(temp_cpds_v)
            #print "v_key:",v_key
            #print "causes: ",temp_caus_vars
            #print "all combine: ",temp_all_comb
            #print "all_prob: ",all_temp_prob_ta
            #print "\n"
    whole_bayes.check_model()
    whole_infer = VariableElimination(whole_bayes)

    for var in latent_var:
        #construct the bayes model within the Markov blanket,Let H denotes the latent variable
        #H's markove blanket includes H,H's causes, H's effect and H's effect's causes
        var_bayes = BayesianModel()
        #H's causes with label
        H_caus = []
        #H's effect with label
        H_effs = []
        #all edges included in the markov blanket, with labels 
        #[(A_1,H_2),...]
        H_edgs = []
        for h_eg in all_edg_label:
            #h's causes (means h is effect)
            if var in h_eg[1]:
                H_caus.append(h_eg[0])
                #H_edgs.append(h_eg)
            #h's effects
            if var in h_eg[0]:
                H_effs.append(h_eg[1])
                #H_edgs.append(h_eg)
        #H's effect's causes (including h)
        H_ef_caus = []
        for h_eg in all_edg_label:
            if h_eg[1] in H_effs:
                H_ef_caus.append(h_eg[0])
        #all variable with H, could be H_1,H_2,... This is important !!!!!!
        H_invars = []
        for h_vit in all_varlabels:
            if var in h_vit:
                H_invars.append(h_vit)
        #all variables within the markov blanket
        tmp_h_mar_lis = H_invars+H_caus+H_effs + H_ef_caus
        markov_vars = []
        markov_vars = list(set(tmp_h_mar_lis))

        #find all edges in the markov blanket
        for h_eg in all_edg_label:
            if h_eg[0] in markov_vars and h_eg[1] in markov_vars:
                H_edgs.append(h_eg)
        #Get all causes and all effects in the markov blanket
        t_h_all_caus=[]
        t_h_all_effs = []
        for h_it in H_edgs:
            t_h_all_caus.append(h_it[0])
            t_h_all_effs.append(h_it[1])
        h_all_caus = []
        h_all_effs = []
        h_all_caus = list(set(t_h_all_caus))
        h_all_effs = list(set(t_h_all_effs))
        #check duplicates
        if len(H_edgs)!=len(list(set(H_edgs))):
            print "H_edges has duplcates: "
            sys.exit(1)
        #Build the temp bayesian model for inferring H
        #add edges
        for h_it in H_edgs:
            var_bayes.add_edge(h_it[0],h_it[1])
        #defining individual CPDs for all origin causes (have no causes themselves)
        #origin causes
        h_origin_caus= []
        for h_it in markov_vars:
            #check whether it is an origin cause and not a latent variable
            if h_it in h_all_caus and h_it not in h_all_effs and h_it not in H_invars:
                #get the origin cause's probability 
                orig_ca_prob = []
                tm_re = (whole_infer.query([h_it]))[h_it]
                #print "h origin causes: ",h_it
                #print "prob: ",tm_re
                #check nan
                if math.isnan(((tm_re.values)[0])):
                    orig_ca_prob.append(1.0)
                    orig_ca_prob.append(0)
                    print "Error:  shouldn't happen: "
                    sys.exit(1)
                else:
                    orig_ca_prob.append((tm_re.values)[0])
                    orig_ca_prob.append((tm_re.values)[1])
                tmp_cpd =TabularCPD(variable=h_it,variable_card=2,values=[[orig_ca_prob[0],orig_ca_prob[1]]])
                var_bayes.add_cpds(tmp_cpd)
                h_origin_caus.append(h_it)
        #defining the individual CPDs for other variables of the variable
        for h_vk in markov_vars:
            if h_vk not in h_origin_caus:
                #get all the causes for this variable
                #storing the causal rela index for all relationships that has h_vk as their effect
                h_re_ind = []
                #all_causes
                h_temp_caus_vars = []
                #cards
                h_temp_cards = []
                #the probability for each relationshp (with labels)
                h_temp_rela_prob = []
                for h_egit in H_edgs:
                    if h_egit[1]==h_vk:
                        h_temp_caus_vars.append(h_egit[0])
                        h_temp_cards.append(2)
                        #find the causal relationships probability
                        #find the index first
                        h_tm_in = 0
                        for in_k,in_v in causal_model_dic.items():
                            if in_v[0] in h_egit[0] and in_v[1] in h_egit[1]:
                                h_tm_in = in_k
                        #store the index
                        h_re_ind.append(h_tm_in)
                        h_temp_rela_prob.append(causal_window_pro_dic[h_tm_in])
                #compute the conditioal probability table for this variable h_vk with labels
                #compute all possible combinations
                h_temp_all_comb = []
                h_fir_comb = []
                for th_it in h_temp_caus_vars:
                    h_fir_comb.append(0)
                h_temp_all_comb.append(h_fir_comb)
                #get the combinations
                for th_it in range(1,len(h_temp_caus_vars)+1):
                    #copy
                    h_temp_it_lis = copy.copy(h_temp_all_comb)
                    for h_iit in range(len(h_temp_it_lis)):
                        h_temp_iit_lis = copy.copy(h_temp_it_lis[h_iit])
                        #modify
                        h_temp_iit_lis[len(h_temp_caus_vars)-th_it]=1
                        #update the raw list
                        h_temp_all_comb.append(h_temp_iit_lis)
                #compute the conditional probability
                h_temp_p_ta = []
                for h_ite in h_temp_all_comb:
                    h_temp_pr_lis = []
                    if sum(h_ite) ==0:
                        h_temp_pr_lis.append(1.0)
                        h_temp_pr_lis.append(0.0)
                    else:
                        #compute the case when the effect has 0 value
                        h_zero_pro = 1.0
                        for hz_it in range(0,len(h_ite)):
                            if h_ite[hz_it]==1:
                                #get the causal relationship's probability
                                h_zero_pro = h_zero_pro*(1.0-causal_window_pro_dic[h_re_ind[hz_it]])
                        h_temp_pr_lis.append(h_zero_pro)
                        h_temp_pr_lis.append(1.0-h_zero_pro)
                    #upadte the whole prob
                    h_temp_p_ta.append(h_temp_pr_lis)
                #transpose
                h_all_temp_prob_ta = (np.asmatrix(h_temp_p_ta)).transpose()
                #add the CPDS
                h_temp_cpds_v = TabularCPD(variable=h_vk,variable_card = 2,values=h_all_temp_prob_ta,evidence = h_temp_caus_vars,evidence_card=h_temp_cards)
                var_bayes.add_cpds(h_temp_cpds_v)
        #CHECK MODEL
        var_bayes.check_model()
        var_infer = VariableElimination(var_bayes)
        print "check_Model success"
        print "H edges: ",H_edgs

        #get the series needs to be inferred
        lat_inf_lis = (infer_data[var]).tolist()
        #Get the causal model and their index of those relationships that are included in our markov blanket, with labels
        #using the same index as causal_model_dic
        h_causal_model_dic = {}
        #e.g. {1:(A_1,B_2),2:....}
        for hm_it in H_edgs:
            for ra_k,ra_v in causal_model_dic.items():
                if ra_v[0] in hm_it[0] and ra_v[1] in hm_it[1]:
                    h_causal_model_dic[ra_k]=hm_it

        #Get the number of latent variables inluded in our markove blanket
        h_latent_var = []
        for h_itv in latent_var:
            for h_iitv in markov_vars:
                if h_itv in h_iitv:
                    h_latent_var.append(h_iitv)
        #initialze the lat_obser_pk
        for h_inv_it in H_invars:
            lat_obser_pk[h_inv_it]=[]
        #print "current latent :",hvr_it
        #start inferring
        for it in range(0,timeseries_leng):
            #inferring for each different latetn for var e.g. H_1,H_2,... H_3
            #the probability result for each latent H_1,H_2,....
            h_resu_dic = {}
            #e.g. {H_1:0.3,H_2:0.4,...}
            for hvr_it in H_invars:
                #suppose to do computation
                whole_count = whole_count + 1

                #initilize the h_resu_dic
                h_resu_dic[hvr_it]=0.0
                #get the latent variables expet the current inferring one 
                h_latent_exp_var = []
                for hla_va in h_latent_var:
                    if hla_va !=hvr_it:
                        h_latent_exp_var.append(hla_va)
                #find the available evidences for current inference
                h_temp_ok_lis = []
                #e.g. [("A_3",1),("B_4",1),..]
                #causal relas that have been checked wether thy have observations 
                h_temp_c_model = {}
                #e.g. {1:(cause,effect)}
                #variables that have been checked whehter they have observations
                h_covered_vars={}
                #e.g.{"A_3":[indexes that has value 1]}
                #e.g.{"A_3":[10,13,15],...}
                #the index of the latetn variable we want to infer
                h_co_lat_lis = []
                h_co_lat_lis.append(it)
                #covered vars starts with the current latent variable and its instance to be one  (the one we want to infer)
                h_covered_vars[hvr_it]=h_co_lat_lis
                #get the number of relationships that in a same causal model, 
                #because for a latent variable with differnt labels they might end up into different models
                inclusive_relas = []
                inclusive_vars = []
                inclusive_vars.append(hvr_it)
                for hh_k,hh_v in h_causal_model_dic.items():
                    if hh_v[0] in inclusive_vars or hh_v[1] in inclusive_vars:
                        inclusive_relas.append(hh_v)
                        inclusive_vars.append(hh_v[0])
                        inclusive_vars.append(hh_v[1])
                        #update inclusive_vars
                        inclusive_vars = copy.copy(list(set(inclusive_vars)))
                inclusive_leng =len(inclusive_relas)
                #print "actual _leng: ",inclusive_leng

                #start looping

                while len(h_temp_c_model.keys())<inclusive_leng:
                    h_temp_covered_vars = {}
                    #find all relationships that connected to the covered vars
                    for h_k,h_v in h_causal_model_dic.items():
                        if h_v[1] in h_covered_vars.keys() and h_v[0] in h_covered_vars.keys() and h_k not in h_temp_c_model.keys():
                            h_temp_c_model[h_k]=h_v
                        #the the target variable is an cause of the covered vars and the target variable is not a latent varialbe iteself
                        #except the current inferring one
                        #this can happend when we have multiple latent variables
                        
                        #the target variable is a cause of the covered ones
                        if h_v[1] in h_covered_vars.keys() and h_v[0] not in h_covered_vars.keys() and h_k not in h_temp_c_model.keys():
                            #update the checked lis
                            h_temp_c_model[h_k]=h_v
                            #the item that h_v[0] has vale 1
                            h_tp_c_lis = []
                            #h_v[0] is a latent variable
                            if h_v[0] in h_latent_exp_var:
                                #update the temp coverd vars
                                h_temp_covered_vars[h_v[0]]=[]
                            else:
                                #checking
                                #get the list for h_v[0]
                                h_tef_la = (h_v[0].split("_"))[0]
                                h_t_ef_lis = (no_noise_data[h_tef_la]).tolist()
                                h_obs_flag = False
                                #check whether has observations
                                h_has_ob_fl = False
                                for h_iit in h_covered_vars[h_v[1]]:
                                    for h_c_it in range(max(0,h_iit-causal_window_dic[h_k][1]),max(0,h_iit-causal_window_dic[h_k][0]+1)): 
                                        h_has_ob_fl = True
                                        if h_t_ef_lis[h_c_it]==1:
                                            h_obs_flag = True
                                            #update h_tp_c-lis
                                            h_tp_c_lis.append(h_c_it)
                                #add observations
                                if h_obs_flag == True:
                                    h_tm_obs_it = (h_v[0],1)
                                    if h_tm_obs_it not in h_temp_ok_lis:
                                        h_temp_ok_lis.append(h_tm_obs_it)
                                if h_obs_flag == False and h_has_ob_fl ==True:
                                    h_tm_obs_it =(h_v[0],0)
                                    if h_tm_obs_it not in h_temp_ok_lis:
                                        h_temp_ok_lis.append(h_tm_obs_it)
                                #update the h_temp_covered_vars
                                if h_v[0] not in h_temp_covered_vars.keys():
                                    h_temp_covered_vars[h_v[0]]=h_tp_c_lis
                                else:
                                    h_temp_covered_vars[h_v[0]]=list(set(h_tp_c_lis+h_temp_covered_vars[h_v[0]]))
                        #the target variable is a effect of the covered vars
                        if h_v[0] in h_covered_vars.keys() and h_v[1] not in h_covered_vars.keys() and h_k not in h_temp_c_model.keys():
                            #update the checked lis
                            h_temp_c_model[h_k]=h_v
                            #the item that h_v[1] has vale 1
                            h_tp_c_lis = []
                            #h_v[1] is a latent variable
                            if h_v[1] in h_latent_exp_var:
                                #update the temp coverd vars
                                h_temp_covered_vars[h_v[1]]=[]
                            else:
                                #checking
                                #get the list for h_v[1]
                                h_tca_la = (h_v[1].split("_"))[0]
                                h_t_ca_lis = (no_noise_data[h_tca_la]).tolist()
                                h_obs_flag = False
                                #check whether has observations
                                h_has_ob_fl = False
                                for h_iit in h_covered_vars[h_v[0]]:
                                    for h_c_it in range(min(timeseries_leng,h_iit+causal_window_dic[h_k][0]),min(timeseries_leng,h_iit+causal_window_dic[h_k][1]+1)):
                                        h_has_ob_fl = True
                                        if h_t_ca_lis[h_c_it]==1:
                                            h_obs_flag = True
                                            #update h_tp_c-lis
                                            h_tp_c_lis.append(h_c_it)
                                #add observations
                                if h_obs_flag == True:
                                    h_tm_obs_it = (h_v[1],1)
                                    if h_tm_obs_it not in h_temp_ok_lis:
                                        h_temp_ok_lis.append(h_tm_obs_it)
                                if h_obs_flag == False and h_has_ob_fl ==True:
                                    h_tm_obs_it =(h_v[1],0)
                                    if h_tm_obs_it not in h_temp_ok_lis:
                                        h_temp_ok_lis.append(h_tm_obs_it)
                                #update the h_temp_covered_vars
                                if h_v[1] not in h_temp_covered_vars.keys():
                                    h_temp_covered_vars[h_v[1]]=h_tp_c_lis
                                else:
                                    h_temp_covered_vars[h_v[1]]=list(set(h_tp_c_lis+h_temp_covered_vars[h_v[1]]))
                    #update covered vars
                    h_covered_vars = copy.copy(h_temp_covered_vars)

                #compute the final k
                fina_pk = 0.0
                h_pk_check_fl = False
                #check whether we already computed the probability
                for h_ob_item in lat_obser_pk[hvr_it]:
                    for h_ob_key,h_ob_val in h_ob_item.items():
                        if set(h_ob_val)==set(h_temp_ok_lis):
                            h_pk_check_fl = True
                            fina_pk = h_ob_key
                if h_pk_check_fl ==True:
                    h_resu_dic[hvr_it]=fina_pk
                else:
                    real_count = real_count +1
                    #compute the pk
                    #compute the evidence
                    h_eviden_dic = {}
                    for h_evi_ite in h_temp_ok_lis:
                        h_eviden_dic[h_evi_ite[0]]=int(h_evi_ite[1])
                    h_re = (var_infer.query([hvr_it],evidence=h_eviden_dic))[hvr_it]
                    #check nan
                    if math.isnan(((h_re.values)[0])):
                        fina_pk = 0.0
                    else:
                        fina_pk = (h_re.values)[1]
                    h_resu_dic[hvr_it]=fina_pk
                    #update the maintained evidence
                    h_temp_ev_dic={}
                    h_temp_ev_dic[fina_pk]=h_temp_ok_lis
                    h_temp_ob_la_lis = []
                    h_temp_ob_la_lis = copy.copy(lat_obser_pk[hvr_it])
                    h_temp_ob_la_lis.append(h_temp_ev_dic)
                    lat_obser_pk[hvr_it]=h_temp_ob_la_lis

            #decide the final pk
            fina_zero = 1.0
            for hf_k,hf_v in h_resu_dic.items():
                fina_zero = fina_zero* (1.0-hf_v)
            fina_one = 1.0-fina_zero
            #infer

            deci = ss.bernoulli.rvs(0.0,0)
            deci = ss.bernoulli.rvs(fina_one,0)
            if deci==1:
                lat_inf_lis[it]=1
        #update the whole series
        infer_data[var]=lat_inf_lis
    #write the whole series
    temp=pd.DataFrame(infer_data)
    print "len(temp)",len(temp)
    infer_data.to_csv(inferseriesfn,index=False)
    print "Whole count: ",whole_count
    print "Real count: ", real_count
    print "Saved percent: ", (float(whole_count)-real_count)/float(whole_count)

if __name__== "__main__":
    if len(sys.argv)!=5:
        print "python infer_latent_series.py paramsfn timeseriesfn nolatentfn inferseriesfn"
        print "fn stands for filename. paramsfn: file storing all the parameters; timeseriesfn: file storing the time series data; nolatentfn: output file storing the time series without the latent variables; inferseriesfn: output file to store the inferred timeseries"
        sys.exit(0)

    paramsfn=sys.argv[1]
    timeseriesfn=sys.argv[2]
    nolatentfn=sys.argv[3]
    inferseriesfn=sys.argv[4]
    load_pickle(paramsfn)
    load_timeseries(timeseriesfn,nolatentfn,inferseriesfn)

