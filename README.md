# Boston House Price Predictor

A machine learning web application that predicts house prices in Boston using a Linear Regression model trained on housing data.

## Features

- **Interactive Web Interface**: Beautiful, responsive design with real-time predictions
- **Machine Learning Model**: Linear Regression trained on Boston housing dataset
- **RESTful API**: Flask backend with JSON API endpoints
- **Real-time Predictions**: Instant price predictions based on property features
- **Mobile Responsive**: Works seamlessly on all devices

## Project Structure

```
boston-house-predictor/
├── app.py                          # Flask application
├── train_model.py                  # Model training script
├── boston_house_price_model.pkl    # Trained model (generated)
├── templates/
│   └── index.html                  # Frontend interface
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Dataset Features

The model uses 13 features to predict house prices:

1. **CRIM**: Crime rate per capita
2. **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft
3. **INDUS**: Proportion of non-retail business acres per town
4. **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. **NOX**: Nitric oxides concentration (parts per 10 million)
6. **RM**: Average number of rooms per dwelling
7. **AGE**: Proportion of owner-occupied units built prior to 1940
8. **DIS**: Weighted distances to employment centers
9. **RAD**: Index of accessibility to radial highways
10. **TAX**: Property tax rate per $10,000
11. **PTRATIO**: Pupil-teacher ratio by town
12. **B**: Proportion of blacks by town
13. **LSTAT**: % lower status of the population

## Installation & Setup

### 1. Clone or Download the Project

```bash
git clone <your-repo-url>
cd boston-house-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Generate the Boston housing dataset
- Train a Linear Regression model
- Save the model as `boston_house_price_model.pkl`
- Display model performance metrics

### 4. Run the Flask Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## API Endpoints

### GET `/`
Returns the main web interface.

### POST `/predict`
Accepts JSON data with property features and returns price prediction.

**Request Body:**
```json
{
  "crim": 0.00632,
  "zn": 18.0,
  "indus": 2.31,
  "chas": 0,
  "nox": 0.538,
  "rm": 6.575,
  "age": 65.2,
  "dis": 4.0900,
  "rad": 1,
  "tax": 296,
  "ptratio": 15.3,
  "b": 396.90,
  "lstat": 4.98
}
```

**Response:**
```json
{
  "prediction": 24.5,
  "status": "success"
}
```

### GET `/health`
Health check endpoint that returns application status.

## Model Performance

The Linear Regression model typically achieves:
- **R² Score**: ~0.85 (85% variance explained)
- **RMSE**: ~$4-6K (Root Mean Square Error)

Key factors affecting house prices:
- Number of rooms (RM) - positive impact
- Crime rate (CRIM) - negative impact
- Lower status population (LSTAT) - negative impact
- Pupil-teacher ratio (PTRATIO) - negative impact

## Usage Examples

### Sample Property Data

You can use this sample data to test the model:

```
Crime Rate: 0.00632
Residential Land: 18.0
Industrial Business: 2.31
Charles River: 0 (No)
NOx Concentration: 0.538
Average Rooms: 6.575
Age of Units: 65.2
Distance to Employment: 4.0900
Highway Access: 1
Property Tax: 296
Pupil-Teacher Ratio: 15.3
Black Population: 396.90
Lower Status Population: 4.98
```

Expected prediction: ~$24K

## Deployment

### Local Development
The application runs on `localhost:5000` by default.

### Production Deployment
For production deployment, consider:
- Using a production WSGI server (e.g., Gunicorn)
- Setting up proper environment variables
- Using a reverse proxy (e.g., Nginx)
- Adding SSL/HTTPS

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (Linear Regression)
- **Data Processing**: NumPy, Pandas
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with modern design
- **Model Persistence**: Pickle

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Troubleshooting

### Common Issues

1. **Model not found error**:
   - Make sure to run `python train_model.py` first
   - Check that `boston_house_price_model.pkl` exists

2. **Import errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Port already in use**:
   - Change the port in `app.py`: `app.run(port=5001)`
   - Or kill the process using port 5000

4. **Prediction errors**:
   - Ensure all form fields are filled with valid numbers
   - Check that feature values are within reasonable ranges

## Future Enhancements

- [ ] Add more sophisticated models (Random Forest, XGBoost)
- [ ] Implement feature scaling and preprocessing
- [ ] Add data visualization and charts
- [ ] Include confidence intervals for predictions
- [ ] Add batch prediction capability
- [ ] Implement user authentication
- [ ] Add prediction history tracking