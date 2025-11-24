# PyHistoPahtomics-Toolkit
This system aims to assist in postoperative ACT decision-making and is implemented through the integration of a pathology-based foundation model, imageomics, machine learning, and Transformer. 

The goal of this repository is:
- to help researchers to reproduce the HPSurv, a histopathomics-based prediction system for individualized overall survival (OS) estimation and ACT optimization.
- to help researchers to build a end-to-end AI model alone to enable fine-grained histological classification, quantification of tumor spatial heterogeneity, OS prediction, and identification of patients most likely to benefit from ACT.
- to provide toolkit for PDAC quantitative analysis and clinical decision support.

## What is HPSurv system?
A histopathomics-based prediction system for individualized overall survival (OS) estimation and ACT optimization. The HPSurv system was developed and tested on 1,020 patients across five clinical centers. It integrated expert annotations, a pathology foundation model, and Transformer architectures to enable fine-grained histological classification, quantification of tumor spatial heterogeneity, OS prediction, and identification of patients most likely to benefit from ACT.
![orig](https://github.com/StandWisdom/PyHistoPahtomics-Toolkit/blob/main/figure/Fig.1.tif)<br>

### Installation
1. [python3 with anaconda](https://www.continuum.io/downloads)
2. [pytorch with/out CUDA](http://pytorch.org)

Using requirements.txt
- Install the required Python packages using pip:
`pip install -r requirements.txt`

Using environment.yml
- Create a Conda environment with the specified dependencies:
`conda env create -f environment.yml`

Activate the Conda environment
- Activate the newly created Conda environment:
`conda activate my_python_env`

### **Step 1: Data Preparation**
- Run preprocessing.py

### **Step 2: Histopathomic Feature Extraction**
- Run ExtractFeatures_main.py

### **Step 3: Feature Filtering**  
- Run Modeling_statistics.py

### **Step 4: Modeling and Prediction**  
- Modeling_ml_surv.py

### **Step 5: Modeling and Survival Prediction**  
- Modeling_ml_surv.py

### **Step 6: Clinical Evaluation**  
- evaluate_pipline.py

---
## Acknowledgement

