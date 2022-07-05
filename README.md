# GATCL2CD
GATCL2CD is effective to predict associations between circRNAs and diseases, which is based on feature convolution learning with heterogeneous graph attention network. The code and details of the paper will be released in this page after the paper accepted.



# Environment Requirement
+ torch == 1.7.1+cu110
+ numpy == 1.19.5
+ matplotlib == 3.5.1
+ dgl-cu110 == 0.5.3



# Model
+ GAT_layer_v2: Coding multi-head dynamic attention mechanism.
+ GATCL.py: the core model proposed in the paper.
+ fivefold_CV.py: completion of a 5-fold cross-validation experiment.
+ case_study.py: get scores for candidate circRNAs for all diseases.




# Compare_models
+ There are five state-of-the-art models including: DMFCDA, CD_LNLP, RWR, GATCDA, IGNSCDA, which are compared under the same experiment settings.

# Question
+ If you have any problems or find mistakes in this code, please contact with me: 
Cheng Yang: yangchengyjs@163.com 
