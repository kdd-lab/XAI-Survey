- Survey Feature removal
colleciton of feature removal methods suggested from reviewer 3 (only 4 citation)


TABULAR

- MAIRE
Extension of Anchor to be applied on continuous
data without the need for binning or discretization by formulating to find the optimal orthotope that explains the prediction of a given test instance

- CERTIFAI
Both for tabular and images (in theory any type, but these two were tested).
They propose a framework for interpretability, fairness and robusteness. It is a model agnostic approach, tailored for any kind of input data, that outputs counterfactual explanations. 
They also introduce CERScore, a metric to evaluate the robusteness of the models without knowing the internals of the model.

- INVASE
Similar to L2X but with different selectors


IMAGES

- Deep taylor Decomposition
general framework of LRP

- minimal image represetation
antenato the metodi di masking (2015)

- Meaningfull Perturbation
Masking method
code: https://github.com/jacobgil/pytorch-explain-black-box

- Extremal Perturbation
Masking method similar to RISE
code: https://github.com/facebookresearch/TorchRay

- FIDO-CA
similar to the masking method but it is replacing blurring woth features draws from a generative model

- Masking Model (2017)
another masking method

-CXPlain
removes single features (or groups of features) for individual inputs and measures the change
in the loss function

- prediction difference analysis
] removes individual features (or groups of features) and analyzes the difference in a model’s predictions (2017)

- RETAIN
Interpretable CNN

- PACE
Concept Based  method similar to ACE

- Leveraging latent features for local explanations
Variation of CEM

- Influence functions
To identify the points (of an image) most representative for a given prediction, this method exploits influence functions ( a concept derived from statistics). They propose an algorithm that approximates the true value of the influence functions, able to work even for non-covex and non-differentiable models. They propose this method to understand the model behaviour and for debugging. They also show how to exploit the information obtained from this method to create an attack dataset for adversarial attacks. 

TEXT

OTHER
- On Evaluating GNN Explanation Methods
Ground truth evalutaion graph NN


