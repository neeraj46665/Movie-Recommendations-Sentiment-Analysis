# Movie Recommendation and Sentiment Analysis App

This repository contains the code for a Streamlit-based movie recommendation app with sentiment analysis features.

## Features

- Recommend movies based on user selection and movie overview similarity.
- Fetch movie posters and details from TMDb API.
- Display reviews for the selected movie and predict their sentiment using a pre-trained model.
- User-friendly interface with dropdowns, images, and color-coded sentiment analysis.

## Requirements

- Python 3.7+
- streamlit
- requests
- pandas
- pickle
- sklearn
- Pillow
- dotenv

## Instructions

1. Clone this repository.
2. Create a file named `.env` in the root directory and add your TMDb API key: `TMDB_API_KEY=<your_api_key>`.
3. Install the required libraries: `pip install -r requirements.txt`.
4. Load the pre-trained sentiment analysis model and movie data: `python load_data.py`.
5. Run the app: `streamlit run app.py`.

## Code Structure

- `app.py`: Main script for running the Streamlit app.
- `utils.py`: Helper functions for fetching data, predicting sentiment, and recommending movies.
- `data`: Folder containing the pre-trained model (`sentiment_model.pkl`) and movie data (`list.pkl`).
- `.env`: File containing the TMDb API key.

## Contribution

Feel free to contribute to this project by suggesting improvements, fixing bugs, or adding new features.

## Disclaimer

The provided TMDb API key is for testing purposes only. Please obtain your own API key for production use.

## Additional Notes

- This README is a template and may need adjustments based on your specific project.
- Consider adding screenshots or GIFs of the app in action.
- Include links to relevant documentation and resources.
