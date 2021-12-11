# SEVulDet #
SEVulDet is a semantics-enhanced deep learning-based framework that can accurately pinpoint vulnerability patterns by preserving path semantics and by learning from arbitrary-length code fragments.

*E-mail* for communication: Provided after the double blind review.

## Details of SEVulDet

Recent years have seen increased attention to deep learning-based vulnerability detection frameworks that leverage neural networks to identify vulnerability patterns. Considerable efforts have been made; still, existing approaches are less accurate in practice. Prior works fail to comprehensively capture semantics from source code or adopt the appropriate design of neural networks.

we presents SEVulDet, a semantics-enhanced deep learning-based framework that can accurately pinpoint vulnerability patterns by preserving path semantics and by learning from arbitrary-length code fragments. More specifically, SEVulDet proposes a path-sensitive code slicing approach for preserving path semantics and logical integrity into code gadgets. Moreover, a spatial pyramid pooling layer is properly added into a convolutional neural network with a multilayer attention mechanism so that SEVulDet can handle code gadgets with flexible length semantics accurately. Experimental results show that SEVulDet vastly outperforms classical static approaches and excels with state-of-the-art deep learning-based solutions, with an average F1-score of up to 94.5%. Notably, the elaborate design of the SEVulDet architecture helped us identify a vulnerability that was not reported by existing techniques.


## Experiment Result

![classical frameworks](PrototypeSystem/img/classical.png)

we compare SEVULDET with the open-source analysis tool Flawfinder, the Rough Auditing Tool for Security (RATS), the commercial detection tool Checkmarx, and the similarity-based framework VUDDY. We observe that our framework SEVULDET vastly outperforms the state-of-the-art classical static vulnerability detection methods.

----

![DL-based frameworks](PrototypeSystem/img/DL-based.png)

SEVULDET excels with state-of-the-art deep learning-based solutions, with an average F1-score of up to 94.5%. 

----

![token heat map](PrototypeSystem/img/token_heat_map.png)

We feed CVE-2016-9776 code gadget into pretrained SEVulDet and visualize the ten tokens of most interest to the attention mechanism. Multiple of these tokens appear on lines 463, 465, 466, and 467, which are the locations where the vulnerability was formed.





