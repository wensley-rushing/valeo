Physically Feasible Semantic Segmentation

Abstract:
State-of-the-art semantic segmentation models are typically optimized in a data-driven fashion, minimizing solely per-pixel classification objectives on their training data. This purely data-driven paradigm often leads to absurd segmentations, especially when the domain of input images is shifted from the one encountered during training. For instance, state-of-the-art models may assign the label ``road'' to a segment which is located above a segment that is respectively labeled as ``sky'', although our knowledge of the physical world dictates that such a configuration is not feasible for images captured by forward-facing upright cameras. Our method, Physically Feasible Semantic Segmentation (PhyFea), extracts explicit physical constraints that govern spatial class relations from the training sets of semantic segmentation datasets and enforces a differentiable loss function that penalizes violations of these constraints to promote prediction feasibility. PhyFea yields significant performance improvements in mIoU over each state-of-the-art network we use as baseline across ADE20K, Cityscapes and ACDC, notably a $1.5\%$ improvement on ADE20K and a $2.1\%$ improvement on ACDC

Installation : 
Follow the steps here
https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation
