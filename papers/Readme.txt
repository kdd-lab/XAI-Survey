TABULAR

- MAIRE
Extension of Anchor to be applied on continuous
data without the need for binning or discretization by formulating to find the optimal orthotope that explains the prediction of a given test instance

- CERTIFAI
Both for tabular and images (in theory any type, but these two were tested).
They propose a framework for interpretability, fairness and robusteness. It is a model agnostic approach, tailored for any kind of input data, that outputs counterfactual explanations. 
They also introduce CERScore, a metric to evaluate the robusteness of the models without knowing the internals of the model.


IMAGES

Variation of gradcam:
- Score-CAM
- Eigen-CAM
- Ablation-CAM

Interpretable CNN
- RETAIN

Variation of Integrated Gradient:
- AGI

Concept Based 
- PACE

Variation of CEM
- Leveraging latent features for local explanations

Influence functions
- To identify the points (of an image) most representative for a given prediction, this method exploits influence functions ( a concept derived from statistics). They propose an algorithm that approximates the true value of the influence functions, able to work even for non-covex and non-differentiable models. They propose this method to understand the model behaviour and for debugging. They also show how to exploit the information obtained from this method to create an attack dataset for adversarial attacks. 

TEXT

OTHER
Ground truth evalutaion graph NN
- On Evaluating GNN Explanation Methods

