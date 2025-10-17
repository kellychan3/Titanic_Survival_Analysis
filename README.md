# 🛳️ Titanic Survival Analysis


**Table of Contents**

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)


## Project Overview

The goal of this project is to **predict the survival of passengers** aboard the Titanic using an **Artificial Neural Network (ANN)** model. The model aims to determine the likelihood that a passenger survived the disaster.

## Dataset

The dataset used in this project `TitanicSurvivalDataNumeric.pkl` contains cleaned and preprocessed numerical data derived from the original Titanic dataset. The target variable, “Survived”, indicates whether a passenger survived (1) or did not survive (0) the sinking of the Titanic.
<img width="824" height="238" alt="image" src="https://github.com/user-attachments/assets/dc050bcf-d7ab-4cce-b7b0-209c18f535fd" />

**Feature Description:**
- Survive: Target variable indicating whether the passenger survived the Titanic disaster. 1 means the passenger survived and 0 means the passenger did not survive.
- Pclass: Passenger class (1st, 2nd, or 3rd), indicating socioeconomic status.
- Sex: Encoded gender of the passenger (0 = male, 1 = female).
- Age: Age of the passenger in years.
- SibSp: Number of siblings or spouses aboard the Titanic.
- Parch: Number of parents or children aboard the Titanic.
- Fare: Ticket fare paid by the passenger.
- Embarked_C: Encoded indicator for passengers who embarked at Cherbourg.
- Embarked_Q: Encoded indicator for passengers who embarked at Queenstown.
- Embarked_S: Encoded indicator for passengers who embarked at Southampton.

## Exploratory Data Analysis
**Target Distribution:** About 38% of passengers survived while 62% did not, indicating an imbalanced dataset. 
<img width="552" height="427" alt="image" src="https://github.com/user-attachments/assets/7e8641ba-53b7-451e-baa6-5ed17784e8ca" />

**Key Insights from the Heatmap**
- Survived vs Pclass (-0.34) → Negative correlation shows that lower-class passengers (higher Pclass number) had lower survival rates.
- Survived vs Sex (-0.54) → Strong negative correlation means that female passengers (encoded as 1) were more likely to survive than males (0).
- Survived vs Age (-0.06) → Very weak negative correlation; age had little influence on survival probability.
- SibSp and Parch (0.41) → Moderate positive correlation; people traveling with siblings/spouses often also had parents/children on board, indicating family groups traveling together.
- Survived vs Fare (0.26) → Positive correlation suggests that passengers who paid higher fares (usually in higher classes) had higher chances of survival.
- Pclass and Fare (-0.55) → Strong negative correlation; first-class passengers paid much higher fares, while third-class paid the least.
- Embarked_C, Embarked_Q, Embarked_S → Strong negative correlations among these features (e.g., -0.78 between C and S) are expected since a passenger can embark from only one port, making these mutually exclusive categorical encodings.
<img width="1133" height="1204" alt="image" src="https://github.com/user-attachments/assets/65ce9a33-bb8c-4e9e-93d7-28f54a1a1085" />

## Model Building
**Architecture:**
- Input layer: 9 features (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q, Embarked_S)
- Hidden layers: 
    - Layer 1: 10 neurons, ReLU activation
    - Layer 2: 6 neurons, ReLU activation
- Output layer: 1 neuron, Sigmoid activation (predict survival probability)
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Training: 100 epochs, batch size = 5 (best hyperparameters from tuning)
<img width="1222" height="423" alt="image" src="https://github.com/user-attachments/assets/550ff543-bea8-4a66-bde7-0a2601ddcd0f" />

## Model Evaluation
**Classification Report:**
- The model achieves 81% accuracy on the test data.
- It predicts non-survivors (class 0) more reliably than survivors (class 1), with higher recall for class 0 (0.89 vs 0.68).
- Precision is balanced across classes, but the model tends to miss some survivors.
- Overall, the model performs reasonably well but could be improved in detecting survivors.
<img width="625" height="197" alt="image" src="https://github.com/user-attachments/assets/680999e2-315f-48f9-b23f-b1f18735ade8" />

**Confusion Matrix:**
- True Positive (76): correctly predicted survivors.
- False Positive (17): predicted as survivors but did not survive.
- False Negative (35): predicted as non-survivors but actually survived.
- True Negative (140): correctly predicted non-survivors.
<img width="513" height="470" alt="image" src="https://github.com/user-attachments/assets/32a9d6b1-786b-4e90-a12a-98301af05a74" />

## Conclusion
The ANN model achieved **81% accuracy** on the test set. It predicts non-survivors more accurately than survivors, as shown by the higher recall for class 0 (0.89 vs 0.68). The confusion matrix indicates the model correctly identified 140 non-survivors and 76 survivors, while misclassifying 17 non-survivors as survivors and 35 survivors as non-survivors. Overall, the model performs well but **could be improved in detecting survivors**, potentially by adjusting thresholds, adding more training epochs or using additional features.
