# LesionXAI – Lesion-Aware Visual Explanation for Apple Leaf Disease Detection

This project uses **Grad-CAM** over a pre-trained **ResNet50** model to generate visual explanations for apple leaf disease classification. It compares these visual outputs with annotated lesion masks using the **Intersection over Union (IoU)** metric to evaluate alignment and model interpretability.

> Built as part of the **Neurostack Internship Task 2025**.

---

---

##  Setup Instructions

### Clone the repository

```bash
git clone https://github.com/shecodes1912/LesionXAI.git
cd LesionXAI

###Install dependencies
pip install -r requirements.txt

###Dataset
 used the annotated leaf disease segmentation dataset available: https://github.com/neeek2303/Leaf-diseases-segmentation

###How to Run the Pipeline
From the root folder, run:


python src/main.py
This will:

Load the dataset
Generate Grad-CAM heatmaps
Threshold to binary masks
Compute IoU with ground truth
Save visualizations to results/visualisations/
Save scores to results/metrics.csv

###Acknowledgment
Special thanks to Neurostack for the opportunity to build this research-grade project and explore explainability in AI. This work further fueled my passion for applying AI to real-world scientific problems.

### License
This project is shared for academic and educational purposes. Feel free to fork, clone, or cite with credit. Dataset belongs to original authors.



