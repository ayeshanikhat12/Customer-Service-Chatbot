# Customer Service Chatbot

##### **Project Type:** NLP / Classification

##### **Industry:** Codec Technologies

##### **Contribution:** Individual

##### **Name:** Ayesha Nikhat

## 1. Project Overview

The **Customer Service Chatbot** is a Machine Learning project that identifies the intent of a customer message using Natural Language Processing (NLP). It uses TF-IDF for text feature extraction and Logistic Regression for intent classification, then returns a predefined guidance response corresponding to the predicted intent.

The project includes dataset preparation, text preprocessing, model training and evaluation, intent analysis, and an interactive web dashboard built using Streamlit.

## 2. Project Objectives

- Develop a Machine Learning model to classify customer messages into supported intents.
- Process customer queries using NLP techniques.
- Convert text into numerical features using TF-IDF.
- Train and evaluate a Logistic Regression classifier.
- Provide relevant predefined guidance based on predicted intents.
- Build an interactive customer support interface using Streamlit.

## Key Features

- **Intent Classification:** Predicts the intent behind customer messages.
- **TF-IDF Vectorization:** Converts text into numerical features for classification.
- **Automated Guidance:** Returns predefined responses mapped to predicted intents.
- **Model Evaluation:** Reports test accuracy and a classification report.
- **Interactive Interface:** Provides a chat interface built with Streamlit.
- **Google Colab Support:** Includes a notebook for training and evaluating the model.

## Dataset

The project uses the Bitext sample customer-service training dataset located at `data/customer_support.csv`.

- **Input feature:** `utterance` — the customer's message.
- **Target label:** `intent` — the intent to predict.
- Rows missing either field are excluded during training.
- Responses are mapped to predicted intents in `src/chatbot.py`, because the dataset does not contain a response column.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset handling |
| TF-IDF | Text feature extraction |
| Logistic Regression | Intent classification |
| Scikit-learn | Machine learning pipeline and evaluation |
| Joblib | Saving and loading model artifacts |
| Streamlit | Interactive chatbot interface |
| Google Colab | Optional notebook-based workflow |

## How It Works

1. A customer enters a message.
2. The trained TF-IDF vectorizer transforms the text into numerical features.
3. The Logistic Regression classifier predicts the message intent.
4. The application maps the predicted intent to a predefined guidance response.
5. The chatbot displays the response in the interface.

## Model Evaluation

The bundled baseline was evaluated using an **80/20 stratified train-test split**.

| Metric | Result |
|---|---|
| Test Accuracy | **99.3%** |

The training script prints the current test accuracy and classification report.

**Important:** This result applies to the reported dataset split. It does not guarantee equivalent performance on real-world customer messages.

## Project Structure

```text
Customer-Service-Chatbot/
├── app.py
├── data/
│   └── customer_support.csv
├── models/
│   ├── chatbot_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── chatbot_model.ipynb
├── src/
│   ├── train_model.py
│   ├── preprocessing.py
│   └── chatbot.py
├── requirements.txt
└── README.md
```

*The structure above reflects the described project files; adjust filenames if your repository differs.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ayeshanikhat12/Customer-Service-Chatbot.git
cd Customer-Service-Chatbot
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train_model.py
```

The training script evaluates the model and saves the trained artifacts under `models/`.

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal by Streamlit.

## Google Colab

To use the notebook workflow:

1. Open `notebooks/chatbot_model.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. Upload the dataset when prompted.
4. Train and evaluate the baseline model.

The notebook saves the classifier and TF-IDF vectorizer.

## Limitations

- The chatbot is limited to the intents and responses supported by the project.
- Responses provide general guidance rather than verified account-specific information.
- The application is not connected to live order, payment, refund, or customer account systems.
- High test accuracy on a sample dataset does not guarantee reliable performance on unseen real-world queries.

## Future Improvements

- Evaluate performance on additional, unseen customer messages.
- Review precision, recall, and F1-score for individual intents.
- Improve handling of unsupported or low-confidence queries.
- Expand the intent and response mappings.
- Explore integration with authorized customer support systems.

## Author

**Ayesha Nikhat**

- GitHub: [ayeshanikhat12](https://github.com/ayeshanikhat12)
- LinkedIn: [Ayesha Nikhat](https://linkedin.com/in/ayesha-nikhat-12a4aa305)

## Project Purpose

This project was developed to demonstrate practical skills in NLP, text feature extraction, machine learning classification, model evaluation, and Streamlit application development.