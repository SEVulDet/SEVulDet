
Model of SEVulDet
---
### Description
**Experimental environment:** python 3.8, pytorch 1.4.0

**attention_token.py:** The content described in Figure 4 and formulas 1 to 4 in the paper. Not every token is equally important for our vulnerability detection, so the purpose of embedding attention is to find out the most significant contributions for vulnerability detection.

**bilstm.py:** We use the network structure used by SySeVR and VulDeePecker for comparison experiments.

**cbam.py:** The content described in formulas 5 to 8 in the paper. We add channel attention and spatial attention from the CBAM to our network to improve the performance of convolution.

**cnn.py:** A convolutional neural network model with only two convolution operations is used for comparison experiments.

**cnn_att_spp.py:** A convolutional neural network model with two convolution operations, attention mechanism, and spatial pooling is the network structure of SEVulDet.

**random_forest.py:** A random forest model for comparison experiments.


###Quick Start
```
python3 main.py ＃CodeGadgetFilePath ＃OutputFileName
```
###Attention Heat Map
```
python3 attention_token.py 10
```