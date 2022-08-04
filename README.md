# License

Copyright (C) 2022 Li Peng (plpeng@hnu.edu.cn), Cheng Yang (yangchengyjs@163.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.



# GATCL2CD
GATCL2CD is effective to predict associations between circRNAs and diseases, which is based on feature convolution learning with heterogeneous graph attention network. 



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
+ If you have any problems or find mistakes in this code, please contact with us: 
Cheng Yang: yangchengyjs@163.com ; Li Peng: plpeng@hnu.edu.cn
