# Breast_cancer_prediction

Sure, let's summarize what you've done using the STAR methodology:

**S - Situation:**
You have worked on building and training a neural network model for binary classification using the breast cancer dataset from sklearn. The goal was to classify whether a tumor is malignant (1) or benign (0) based on features extracted from diagnostic images.

**T - Task:**
1. **Data Preparation:**
   - Loaded the breast cancer dataset and split it into training and test sets.
   - Standardized the data using `StandardScaler` to ensure all features have a similar scale.

2. **Model Building:**
   - Defined a neural network architecture (`NeuralNet`) using PyTorch, consisting of two fully connected layers with ReLU activation between them and a sigmoid activation function for binary classification.

3. **Model Training:**
   - Moved data to GPU (if available) and initialized the neural network.
   - Defined hyperparameters including learning rate, number of epochs, and model size.
   - Used Binary Cross Entropy Loss (`nn.BCELoss()`) and Adam optimizer (`optim.Adam`) for training.
   - Iteratively trained the model on the training data, computing loss, and monitoring accuracy.

4. **Model Evaluation:**
   - Evaluated the model's performance on both training and test sets.
   - Calculated accuracy metrics to assess how well the model predicts tumor types.

**A - Action:**
You implemented the code for data preprocessing, model definition, training loop, and evaluation metrics calculation. You used PyTorch for building and training the neural network, ensuring compatibility with CUDA for GPU acceleration if available.

**R - Result:**
1. Achieved an accuracy of **97.36%** on the training data after 100 epochs.
2. Achieved an accuracy of **96.49%** on the test data, demonstrating that the model generalized well to unseen data.

In conclusion, you successfully built and trained a neural network model using PyTorch for binary classification of breast cancer tumors based on diagnostic features. The model demonstrated high accuracy on both training and test datasets, indicating its effectiveness in predicting tumor malignancy.
