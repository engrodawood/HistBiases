# HistBiases: Predicting Omics-Based Biomarkers from Histological Images

This repository contains the data files, model predictions, and code scripts used in the study: **"Buyer Beware: Confounding Factors and Biases Abound When Predicting Omics-Based Biomarkers from Histological Images."** The study critically examines the limitations of current deep learning models in computational pathology, particularly in predicting genomic, transcriptomic, and molecular biomarkers from routine histology whole slide images (WSIs).

## Overview

Recent advances in computational pathology have introduced deep learning methods to predict biomarkers from histological images for cancer diagnosis, prognosis, and treatment. However, these methods often overlook the co-dependencies among biomarkers, leading to confounded predictions. Our study investigates these interdependencies and their impact on model predictions.

### Key Findings

- **Significant Interdependencies:** Statistical analysis reveals substantial interdependencies among biomarkers, influenced by biological processes and sampling artifacts.
- **Confounded Predictions:** Current models tend to predict the aggregated influence of biomarkers rather than isolating individual effects, leading to confounded predictions.
- **Limited Advantages of Deep Learning:** Due to these interdependencies, deep learning models may not offer significant improvement over traditional pathologist-examined features, such as grade, in predicting certain biomarkers.

### Methods
![Workflow Diagram](https://github.com/user-attachments/assets/ccc38b3f-bb17-4b8d-92a5-302ef5635d68)

Diagram illustrates conceptual framework of ML methods aimed at predicting the status of molecular biomarkers from histology images. 
A) Machine learning-based prediction of molecular characteristics or omics biomarkers from whole slide images involves using training data of WSIs with known biomarker statuses. The ML model accepts the representation of the whole slide image (X) as input and predicts the status of a certain biomarker (Y) as the target. B) An ideal predictor should be able to predict mutational or omics-based biomarker signatures using features based on the histological effects of that biomarker, and its output (Z) should be independent of unrelated confounding factors (lumped into a variable C) as shown in the simplified causal diagram. Conversely, if the predictor’s output is dependent upon the histological effects of (Y) as well as other confounding factors such as histological grade or tumor mutational burden, it may not be possible to tease out the individual effect of Y independently.

### Interpretation

Our findings highlight the need to revisit model training protocols in computational pathology to account for biomarker interdependencies. This includes selecting diverse datasets, defining prediction variables based on co-dependencies, designing models to disentangle complex relationships, and conducting stringent stratification testing. Failure to address these factors may lead to suboptimal clinical decisions.

## Repository Contents

- **Data Files:** Datasets used in the study.
- **Model Predictions:** Predictions generated by the machine learning models.
- **Code Scripts:** Scripts used for model training, testing, and analysis.

## Citation

If you use this repository, please cite our preprint:

> [Buyer Beware: Confounding Factors and Biases Abound When Predicting Omics-Based Biomarkers from Histological Images](https://www.biorxiv.org/content/10.1101/2024.06.23.600257v1)

## License

The source code of SlideGraph∞ is released under the MIT-CC-Non-Commercial license. For CLAM, please refer to the official repository for licensing information: [CLAM GitHub Repository](https://github.com/mahmoodlab/CLAM/tree/master).

This project is licensed under the MIT License. See the LICENSE file for details.

---

For more information, please refer to the [preprint](https://www.biorxiv.org/content/10.1101/2024.06.23.600257v1).
