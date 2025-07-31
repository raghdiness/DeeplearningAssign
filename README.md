# Fashion MNIST Classifier

This project is a deep learning assignment focused on building and evaluating neural networks using the Fashion MNIST dataset. The dataset includes grayscale images of clothing items, with the goal of classifying them into one of 10 categories.

## Dataset: Fashion MNIST

Fashion MNIST is a popular benchmark dataset for image classification. It contains:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 clothing categories (e.g., T-shirt, trousers, pullover, dress)

## Project Goals

- Preprocess and visualize the dataset
- Train and evaluate:
  - A simple dense neural network
  - A model using dropout for regularization
  - A convolutional neural network (CNN)
- Compare performance of different models

## Workflow

1. **Data Loading and Exploration**
   - Load dataset from TensorFlow Keras
   - Display sample images and labels

2. **Data Preprocessing**
   - Normalize pixel values to [0, 1]
   - Flatten images for dense networks
   - Reshape images for CNN input (if applicable)

3. **Model Building**
   - Dense NN with 2 hidden layers
   - Dropout layer to reduce overfitting
   - CNN using Conv2D, MaxPooling, and Flatten

4. **Training and Evaluation**
   - Compile models with appropriate loss functions and optimizers
   - Use accuracy as evaluation metric
   - Visualize training history (loss & accuracy curves)

5. **Testing**
   - Evaluate on the test set
   - Display predictions on sample images

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Skills Demonstrated

- Neural network architecture design
- Image normalization and reshaping
- Use of dropout and convolution layers
- Model evaluation and tunin
