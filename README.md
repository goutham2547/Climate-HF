# ClimateDiscourseNet: Text Classification for Climate Discourse

This repository contains a specialized machine learning pipeline designed to categorize climate change-related quotes into specific classes of skepticism or relevance. The system uses a deep learning approach, leveraging high-dimensional embeddings and a Multi-layer Perceptron (MLP) architecture.

---

## ## Project Overview

The core objective is to identify and categorize various forms of climate discourse, ranging from factual relevance to specific claims regarding fossil fuels or scientific reliability. 

The model classifies text into 8 distinct categories:
* **0**: Not relevant
* **1**: Not happening
* **2**: Not human-caused
* **3**: Not bad
* **4**: Solutions are harmful/unnecessary
* **5**: Science is unreliable
* **6**: Proponents are biased
* **7**: Fossil fuels are needed

---

## ## Technical Architecture

* **Feature Extraction**: The pipeline utilizes the `sentence-t5-large` model to transform raw text into 768-dimensional vector representations.
* **Neural Network**: A custom PyTorch `ClimateDiscourseNet` class is implemented, featuring four linear layers with ReLU activation and Dropout (0.15) for regularization.
* **Optimization**: The training process employs the **AdamW** optimizer with a learning rate of $3e-4$ and a `ReduceLROnPlateau` scheduler to refine weights when validation loss plateaus.
* **Imbalance Handling**: To manage skewed data, the script automatically calculates penalty weights for the `CrossEntropyLoss` function based on class frequency in the training set.

---

## ## Requirements

To run the classification pipeline, ensure the following Python libraries are installed:
* `torch`
* `numpy`
* `scikit-learn`
* `tqdm`
* `datasets` (Hugging Face)
* `sentence-transformers`
* `huggingface_hub`

---

## ## Usage Instructions

1. **Authentication**: Update the `AUTH_TOKEN` variable in the script with your Hugging Face access token to enable model exporting.
2. **Dataset Configuration**:
    * Set `USE_FULL_COLLECTION = False` to maintain a standard train/test split for development.
    * Set `USE_FULL_COLLECTION = True` to train on the entire available dataset for final production weights.
3. **Execution**: Running the script will automatically download the `quotaclimat/frugalaichallenge-text-train` dataset, encode the quotes, and execute the training cycle.
4. **Output**: The best-performing model parameters will be saved to the `./climate_analysis_model` directory.

---

## ## Performance Monitoring

The script provides real-time progress updates via `tqdm` and logs the accuracy score whenever a new performance peak is reached during the validation phase.
