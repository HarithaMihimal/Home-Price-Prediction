
# Home Price Prediction with Machine Learning

## Overview

This project is a comprehensive machine learning application that predicts home prices based on various features such as location, size, number of rooms, and bathrooms. The project includes a user-friendly interface that allows users to input these features and get an estimated home price.

## Features

- **Machine Learning Model:** The project utilizes a machine learning model trained on a dataset of home prices. The model has been serialized and saved using `pickle` for efficient loading and prediction.
- **Flask API:** The backend server is built using Flask, which serves the model predictions via REST API endpoints.
- **Frontend:** The user interface is built with HTML, CSS, and JavaScript. It allows users to input the relevant features and get the predicted home price.
- **Data Preprocessing:** The data used in the model is processed to ensure accurate predictions, including handling categorical data such as location.
- **Model Artifacts:** The trained model and necessary columns are stored in the `artifacts` directory and loaded during runtime for predictions.

## Project Structure

```
HomePricePredict/
│
├── client/
│   ├── app.html
│   ├── app.css
│   └── app.js
│
├── server/
│   ├── artifacts/
│   │   ├── columns.json
│   │   └── home_prices_model.pickle
│   ├── server.py
│   └── util.py
│
└── venv/
```

- **client/**: Contains the frontend files (HTML, CSS, JavaScript).
- **server/**: Contains the Flask server and utilities for handling predictions.
- **artifacts/**: Stores the model and necessary data files.
- **venv/**: The virtual environment for managing dependencies.

## How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/HomePricePredict.git
   cd HomePricePredict
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server:**
   ```bash
   cd server
   python server.py
   ```

5. **Open the frontend:**
   Open `client/app.html` in your web browser to use the user interface for predicting home prices.

## Dataset

The dataset used for training the model is sourced from Kaggle and contains home prices from Bengaluru, India. You can find the dataset [here](https://www.kaggle.com/datasets/mohitpratap166/bengaluru-house-data).

## Acknowledgments

This project is inspired by the need to provide accessible and accurate home price predictions for individuals looking to buy or sell property. The dataset used for training the model is based on public data from Kaggle.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
![image](https://github.com/user-attachments/assets/38fb468d-bf82-4ef9-a560-2fb3d7b3b820)

