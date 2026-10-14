**House Price Predictor**
An XGBoost regression model trained on the California Housing dataset and used to predict on new dataset. User can select parameters’ values and then click on predict house price to get the estimated price in $.
This was built for the TechCrush C8 AI/ML Class to demonstrate how to build and deploy a ML model.

**What it does**
•	I uses 8 features of in the California dataset
•	Applies the same preprocessing used during training (winsorization + scaling)
•	Returns an estimated median house value in US dollars

**Required Files**
File	Purpose
app.py	The Streamlit app that loads the artifacts and predicts
best_model.json	The trained XGBoost model
scaler.pkl	The fitted StandardScaler
winsor_cap.pkl	Winsorization limits used during training
meta.pkl	Feature names and test metrics
requirements.txt	Packages needed to run the project

**Run it locally**
# 1. Install packages
pip install -r requirements.txt

# 2. Start the app
streamlit run app.py

**Deploy to Streamlit Community Cloud**
1.	Push this entire folder to a public GitHub repository.
Make sure you include:
o	best_model.json
o	scaler.pkl
o	winsor_cap.pkl
o	meta.pkl
2.	Go to https://share.streamlit.io and sign in with GitHub.
3.	Click Create app, select your repository, the main branch, and app.py.
4.	Click Deploy and wait about two minutes.

**Three important lessons**
1.	Train once, load many times
The model is trained and saved. The app only loads the result using Streamlit app.
2.	Save the scaler (and any other preprocessing).
The model learned from scaled and capped numbers. If new data arrives unprocessed, the prediction is meaningless.
3.	@st.cache_resource matters
Streamlit re-runs the whole script on every interaction.
Without the cache decorator the model would be loaded from disk every time.

Model Performance (test set)
•	R² Score ≈ 0.85
•	MAE ≈ 0.28
•	RMSE ≈ 0.44
________________________________________
Made for TechCrush C8 AI/ML Class

