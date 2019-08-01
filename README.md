# Using Domain Knowledge to Overcome Latent Variables in Causal Inference from Time Series

## How to cite  

M. Zheng and S. Kleinberg. Using Domain Knowledge to Overcome Latent Variables in Causal Inference from Time Series. In: *MLHC*, 2019  



## Preparation:

Create a pickle file with the following dictionary; Assume all of your parameters are in a dictionary named mydata:  
        mydata['timeseries\_leng']  
        mydata['causal\_model\_dic']  
        mydata['causal\_window\_dic']  
        mydata['causal\_window\_pro\_dic']  

### the length of the new time series  
timeseries\_leng = 5000  

### dictionary storing causal relationships in our causal model.   
For latent variable scenario, this includes the prior knowledge and causal model
{causal relationship tag: causal relationship}  
1. A->B  
2. A->C  
causal\_model\_dic = {1:["A","B"],2:["A","C"],...}  

### dictionary storing time windows for each causal relationship  
{causal relationship tag: [window\_start,window\_end]}  
Causal Relationship A->B (1) occurs between 2 and 3 timepoints away (inclusive)  
Causal Relationship A->C (2) occurs between 5 and 10 timepoints away (inclusive)  
causal\_window\_dic = {1:[2,3],2:[5,10],...}

### dictionary storing the probability for each time window: 
{causal relationship  tag: pro}  
Causal Relationship A->B (1) is 90%  
Causal Relationship A->C (2) is 85%  
causal\_window\_pro\_dic = {1:0.90,2:0.85,...}  

##  Python Example for Preparation:    

```python
mydata={}  
mydata['timeseries_leng']=timeseries_leng  
mydata['causal_model_dic']=causal_model_dic  
mydata['causal_window_dic']=causal_window_dic  
mydata['causal_window_pro_dic']=causal_window_pro_dic  
paramsfn="/data/params.pkl"  
pickle.dump(mydata,open(paramsfn, "wb"))  
```  

## timeseriesfn Example:

Each column is a variable and each row is a timestep of whether it happened or not. Will be imported into a Pandas dataframe.   

0 means didn't occur, 1 means occured   

A: No, No, Yes   
B: No, Yes, Yes   
C: Yes, No, Yes   

timeseries.csv   
A B C   
0 0 1   
0 1 0   
1 1 1   
...

## Instructions for running:  

fn stands for filename   
paramsfn: file storing all the parameters (above example)   
timeseriesfn: file storing the time series data   
nolatentfn: output file storing the time series without the latent variables   
inferseriesfn: output file to store the inferred timeseries   
```python   
python infer_latent_series.py paramsfn timeseriesfn nolatentfn inferseriesfn  
```   

The file reconstructs the latent variables, then it generates the timeseries, and finally it does conditional independence tests and outputs the answer to inferseriesfn (will be generated along with nolatentfn)    


