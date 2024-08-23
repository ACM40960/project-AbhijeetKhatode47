# Automated Risk Management
![Python](https://img.shields.io/badge/Python-v3.10%2B-blue)
![Tensorflow](https://img.shields.io/badge/Tensorflow-Latest-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-blue)
![Numpy](https://img.shields.io/badge/Numpy-Latest-blue)
![SearBorn](https://img.shields.io/badge/Seaborn-0.13.2-green)
![Random Forest](https://img.shields.io/badge/Classifier-RandomForest-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)
![GitHub Repo stars](https://img.shields.io/github/stars/ACM40960/project-bhupendrachaudhary08?style=social)


## Table of Contents

1. [Abstract](#abstract)
2. [Project Description](#project-description)
   - [Key Components](#Key-Components)
   - [Data Sources](#Data-Sources)
   - [Objectives](#Objectives)
   - [Scope of the Analysis](#Scope-of-the-Analysis)
   - [Significance of the Study](#Significance)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Steps for Installation](#steps-for-installation)
   - [Installation Notes](#installation-notes)
5. [Model Training](#model-training)
   - [Step 1: Train the Model](#step-1-train-the-model)
   - [Step 2: Evaluate the Model](#step-2-evaluate-the-model)
6. [Real-Time Interpretation](#real-time-interpretation)
   - [Step 1: Run the Interpretation Script](#step-1-run-the-interpretation-script)
   - [Step 2: Interact with the System](#step-2-interact-with-the-system)
7. [Results](#results)
   - [Key Metrics](#key-metrics)
10. [Future Work](#future-work)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contact](#contact)
14. [Credits](#credits)

## Abstract

This project develops an automated risk management system for investment portfolios using Long Short-Term Memory (LSTM) networks. By processing historical time-series data, such as stock prices and trading volumes, the system predicts future risks and evaluates their impact. It incorporates advanced preprocessing, LSTM modeling, and regularization techniques to enhance prediction accuracy and mitigate overfitting. The system generates comprehensive risk reports and visualizations, improving efficiency and precision in managing financial risks and supporting better investment decisions.

## Project Description

### Key Components:
- **Data Preprocessing:** Cleans and normalizes financial time-series data to prepare it for modeling, including handling missing values and scaling features.
- **LSTM Model:** Utilizes Long Short-Term Memory networks to capture complex temporal patterns and predict future risks based on historical data.
- **Risk Detection and Assessment:** Identifies potential risks and evaluates their severity using model predictions, generating actionable insights.
- **Risk Mitigation:** Implements strategies based on risk assessments to reduce exposure and manage portfolio risk.
- **Reporting and Visualization:** Produces detailed risk reports and visualizations to communicate findings and support decision-making.

### Data Sources
The data for this project is collected from the following sources:
- **Yahoo Finance:** Historical stock prices, financial metrics, and other market data.
- **Quantstats:** Provides performance metrics and risk analysis for financial data.
- **Pandas DataReader:** Used for pulling data from various remote data sources into a pandas DataFrame.


###Objective:

Enhance Risk Detection: Improve the ability to identify potential risks in investment portfolios using advanced machine learning techniques.
Accurate Risk Assessment: Provide precise evaluations of risk severity and probability based on historical and real-time data.
Effective Risk Mitigation: Develop and implement strategies to proactively manage and reduce identified risks.
Comprehensive Reporting: Generate detailed reports and visualizations to communicate risk findings and support informed decision-making.
Improve Efficiency and Consistency: Automate risk management processes to reduce manual effort and ensure uniform application of risk management rules.

### Supported Gestures

The system is trained to recognise the following ASL gestures:

**Alphabets:**

- A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y

**Numbers:**

- One, Two, Three, Four, Five, Six, Seven, Eight, Nine

![Signs](images/signs.jpg)

## Features

- **Real-time Gesture Recognition**: Detects and interprets ASL gestures using a webcam.
- **Easy Dataset Collection**: Includes scripts for capturing and labeling gesture images.
- **Customisable Model**: Users can extend the model to recognise additional gestures.
- **Performance Visualisation**: Displays metrics like confusion matrices, ROC, and Precision-Recall curves.

## Project Structure

```plaintext
risk_management_project/
├── data/                          # Directory containing raw and processed data files
│   ├── raw/                       # Subdirectory for raw data files
│   ├── processed/                 # Subdirectory for processed data files
├── notebooks/                     # Jupyter notebooks for exploratory analysis and model development
│   ├── EDA.ipynb                  # Notebook for Exploratory Data Analysis
│   ├── risk_analysis.ipynb        # Notebook for risk and return analysis
│   ├── portfolio_optimization.ipynb  # Notebook for portfolio optimization
│   ├── forecasting.ipynb          # Notebook for time series analysis and forecasting
├── models/                        # Directory to save trained models
├── reports/                       # Generated reports and visualizations
│   ├── figures/                   # Subdirectory for figures and plots
├── src/                           # Source code for the project
│   ├── config.py                  # Configuration file with paths and constants
│   ├── data_preprocessing.py      # Script for data cleaning and preprocessing
│   ├── feature_engineering.py     # Script for feature engineering
│   ├── risk_analysis.py           # Script for risk and return calculations
│   ├── portfolio_optimization.py  # Script for portfolio optimization
│   ├── forecasting.py             # Script for time series forecasting
│   ├── visualization.py           # Script for creating visualizations
│   ├── utils.py                   # Utility functions
├── tests/                         # Directory for unit tests
│   ├── test_data_preprocessing.py # Unit tests for data preprocessing
│   ├── test_risk_analysis.py      # Unit tests for risk analysis
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files and directories to ignore in git
└── README.md                      # Project documentation

```

## Installation
   - [Prerequisites](#prerequisites)
   - [Steps for Installation](#steps-for-installation)
   - [Installation Notes](#installation-notes)

### Prerequisites
   - Python 3.x
   - pip (Python package installer)
   - A virtual environment tool (optional but recommended)
   - Git (for version control)

### Steps for Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ACM40960/project-AbhijeetKhatode47/risk_management_project.git
   cd risk_management_project

2. **Create a virtual environment:**

    - **On macOS/Linux:**
     ```sh
     python3 -m venv venv
     source venv/bin/activate
    ```
   - **On Windows:**
     ```sh
     python -m venv venv
     venv\Scripts\activate
     ```

4. **Install the dependencies:**
   ```sh
    pip install -r requirements.txt
   ```

### Installation Notes
1. Make sure your Python version is compatible with the packages listed in requirements.txt.
2. It's recommended to use a virtual environment to avoid dependency conflicts.
3. If you encounter any issues during installation, check the compatibility of your environment or refer to the package documentation.
4. You may need to install additional packages if you extend the project beyond the current scope.

## Dataset Collection

### 1. Data Sources
The data for this project is collected from the following sources:
- **Yahoo Finance:** Historical stock prices, financial metrics, and other market data.
- **Quantstats:** Provides performance metrics and risk analysis for financial data.
- **Pandas DataReader:** Used for pulling data from various remote data sources into a pandas DataFrame.

### 2. Data Collection Process

#### 2.1. Importing Libraries
To start collecting data, the following libraries are imported:

```python
import yfinance as yf
import pandas_datareader.data as web
import datetime as dt
import pandas as pd
import quantstats as qs
```

## Data Description

The collected dataset includes the following fields:
- **Date:** The trading date.
- **Open:** The opening price of the stock on that date.
- **High:** The highest price of the stock on that date.
- **Low:** The lowest price of the stock on that date.
- **Close:** The closing price of the stock on that date.
- **Adj Close:** The adjusted closing price (after accounting for dividends and splits).
- **Volume:** The number of shares traded on that date
    
### Data Validation
After collecting the data, it is crucial to validate it by checking for:
- Missing Values: Ensure there are no missing or NaN values in critical fields like 'Close' and 'Volume'.
- Data Consistency: Verify that the data covers the specified date range without any gaps.
- Correct Tickers: Ensure the correct stock ticker is used for fetching data.

### Notes on Data Collection
- Frequency: Daily stock data is collected, but it can be adjusted (e.g., weekly, monthly) depending on the analysis needs.
- Data Quality: Ensure that data sources are reliable, as inaccurate data can lead to misleading conclusions.
- Updating the Dataset: Regularly update the dataset to include the latest available data for continuous analysis.

## Model Training

### Step 1: Train the Model

To train the Random Forest model on the processed dataset, run the following script:

```bash
python src/model_training.py
```

The script performs the following steps:

- **Splits the Data:** Separates the dataset into training and testing subsets.
- **Model Training:** Trains a RandomForest classifier.
- **Model Evaluation:** Evaluates the model using metrics such as accuracy, confusion matrices, ROC curves, and Precision-Recall curves.
- **Model Saving:** Saves the trained model to the `artifacts/` directory.

### Step 2: Evaluate the Model

During training, the following plots are generated to assess the model's performance:

#### Confusion Matrix:

![Confusion Matrix](images/confusion-matrix.png)

#### ROC and Precision-Recall Curves:

![ROC Curve](images/roc-curve.png)

## Real-Time Interpretation

### Step 1: Run the Interpretation Script

Once the model is trained, run the following script to start real-time gesture recognition:

```bash
python src/app.py
```

### Step 2: Interact with the System

- The script uses your webcam to detect hand gestures in real-time.
- **Confirm Letters:** Press the `spacebar` to confirm a detected letter and add it to the sentence.
- **Create Sentences:** The system allows you to construct sentences by confirming individual letters.
- **Delete the Last Confirmed Letter:** If you make a mistake, you can delete the last confirmed letter by pressing the `B` key.
- **Add Space:** Press the `S` key to add a space between words.

[//]: # "#### Example Video:"
[//]: # "Add a video here showcasing the real-time gesture recognition in action."

## Results

The trained model successfully recognises the following ASL gestures:

- **Alphabets:** A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y
- **Numbers:** One, Two, Three, Four, Five, Six, Seven, Eight, Nine

### Key Metrics:

- **Accuracy:** 100% on the test set.
- **AUC:** 1.00 for all gestures.
- **Precision-Recall:** 1.00 for all gestures.

## Future Work

Future improvements to this project include:

- **Expanding the Gesture Set:** Adding support for more complex gestures, two-handed gestures, and dynamic gestures involving motion.
- **Improving Generalisation:** Collecting a larger, more diverse dataset to improve model robustness in different lighting conditions and environments.
- **Integrating with Other Applications:** Developing a mobile or web application to make the system more accessible in real-world scenarios.

## Contributing

Contributions are welcome! If you'd like to improve this project, please fork the repository and submit a pull request. Your contributions could include adding new features, improving documentation, or fixing bugs.

### Steps to Contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or suggestions, please open an issue or contact me at [bhupendra.chaudhary@ucdconnect.ie](mailto:bhupendra.chaudhary@ucdconnect.ie).

## Credits

This project is in collaboration with [Sahil Chalkhure](https://github.com/ACM40960/project-sahilchalkhure26)
