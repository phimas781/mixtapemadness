import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt # For more interactive plots if needed, but starting with matplotlib/seaborn

# --- 1. Dashboard Configuration ---
st.set_page_config(
    page_title="Song Popularity Predictor Dashboard",
    page_icon="🎶",
    layout="wide", # Use wide layout for better visualization space
    initial_sidebar_state="expanded"
)

# --- 2. Load Assets (Cached for Efficiency) ---
@st.cache_resource # Cache the model and preprocessor to avoid reloading on every rerun
def load_model_assets():
    try:
        model_pipeline = joblib.load('song_popularity_model.joblib')
        model_features = joblib.load('model_features.joblib')
        return model_pipeline, model_features
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'song_popularity_model.joblib' and 'model_features.joblib' are in the same directory.")
        st.stop() # Stop the app if crucial files are missing

@st.cache_data # Cache the historical data to avoid reloading on every rerun
def load_historical_data():
    try:
        df = pd.read_csv('spotify_songs_data.csv')
        df['release_date'] = pd.to_datetime(df['release_date'])
        return df
    except FileNotFoundError:
        st.error("Historical data file 'spotify_songs_data.csv' not found. Please ensure it's in the same directory.")
        st.stop() # Stop the app if crucial files are missing

model_pipeline, model_features = load_model_assets()
historical_df = load_historical_data()

# Get unique artist names for selection
all_artist_names = historical_df['artist_name'].unique().tolist()
overall_avg_popularity = historical_df['track_popularity'].mean()


# --- 3. Helper Functions (Copied from previous Colab cell) ---

def get_artist_historical_metrics(artist_name_input, historical_df):
    """
    Calculates historical metrics for a given artist from the full dataset.
    This function would typically run once when the artist is selected in Streamlit.
    """
    artist_data = historical_df[historical_df['artist_name'] == artist_name_input]
    if not artist_data.empty:
        avg_popularity = artist_data['track_popularity'].mean()
        num_releases = len(artist_data)
        # Get the current (latest) artist followers and popularity snapshot
        # These are snapshots, so picking the latest is reasonable
        latest_artist_info = artist_data.sort_values('release_date', ascending=False).iloc[0]
        artist_followers = latest_artist_info['artist_followers']
        artist_popularity_snapshot = latest_artist_info['artist_popularity_snapshot']
        # Also determine the most common genre for the artist for consistent input
        # Note: We are using the 'artist_genres' column directly from the dataframe here,
        # which is already a joined string from our previous processing.
        # If an artist has multiple genres, we take the mode of the primary genre.
        # For simplicity, we assume one dominant genre for the artist.
        # This will be used as 'artist_genres_cleaned' for prediction.
        artist_genre_cleaned = artist_data['artist_genres'].apply(lambda x: x.split(',')[0].strip() if x and x != 'None' else 'Unknown').mode()[0]

        return {
            'avg_popularity': avg_popularity,
            'num_releases': num_releases,
            'artist_followers': artist_followers,
            'artist_popularity_snapshot': artist_popularity_snapshot,
            'artist_genre_cleaned': artist_genre_cleaned
        }
    return None # Artist not found in historical data

def classify_popularity_level(popularity_score):
    """Classifies a popularity score into success categories."""
    if popularity_score >= 80:
        return "Potential Blockbuster 🌟"
    elif popularity_score >= 60:
        return "Likely Hit 🔥"
    elif popularity_score >= 40:
        return "Moderate Success 👍"
    else:
        return "Niche Appeal 🎧"

def predict_song_popularity_enhanced(artist_name, album_type, release_date_str, duration_ms, explicit,
                                    historical_df, model_pipeline, model_features):
    """
    Predicts the Spotify popularity of a new song with enhanced metrics.

    Args:
        artist_name (str): Name of the artist (Gwamz, KiLLOWEN, or other known artist).
        album_type (str): Type of album ('album', 'single').
        release_date_str (str): Release date in 'YYYY-MM-DD' format.
        duration_ms (int): Duration of the track in milliseconds.
        explicit (bool): True if explicit, False otherwise.
        historical_df (pd.DataFrame): The DataFrame containing historical song data.
        model_pipeline (sklearn.pipeline.Pipeline): The trained scikit-learn pipeline.
        model_features (list): List of feature names the model was trained on.

    Returns:
        dict: Contains predicted popularity, confidence interval, success level, and feature importances.
    """
    # 1. Prepare fixed artist-level inputs (fetch from historical data)
    artist_metrics = get_artist_historical_metrics(artist_name, historical_df)
    if artist_metrics is None:
        st.warning(f"Artist '{artist_name}' not found in historical data. Using overall average/dummy values for prediction context.")
        overall_avg_popularity = historical_df['track_popularity'].mean() if not historical_df.empty else 50
        # For num_prev_releases for an unknown artist, we'll use a placeholder like 0 or 1 for simplicity
        overall_num_releases = 0 # Assume no previous releases for unknown artists for this context
        overall_avg_followers = historical_df['artist_followers'].mean() if not historical_df.empty else 100000
        overall_avg_artist_pop = historical_df['artist_popularity_snapshot'].mean() if not historical_df.empty else 50
        artist_genre_cleaned = 'Unknown'

        artist_followers = overall_avg_followers
        artist_popularity_snapshot = overall_avg_artist_pop
        prev_avg_popularity = overall_avg_popularity
        num_prev_releases = overall_num_releases
    else:
        artist_followers = artist_metrics['artist_followers']
        artist_popularity_snapshot = artist_metrics['artist_popularity_snapshot']
        prev_avg_popularity = artist_metrics['avg_popularity']
        num_prev_releases = artist_metrics['num_releases']
        artist_genre_cleaned = artist_metrics['artist_genre_cleaned']

    # 2. Derive new song specific features
    release_date = pd.to_datetime(release_date_str)
    current_date = pd.to_datetime(datetime.now().date())

    days_since_release = (current_date - release_date).days
    release_year = release_date.year
    release_month = release_date.month
    release_day_of_week = release_date.dayofweek
    explicit_numeric = int(explicit)

    available_markets_count = 180 # A common number for major global releases, or make it an input


    # 3. Create DataFrame for prediction - ensure column order matches model_features
    # Fill in `artist_genres_cleaned` based on `artist_name` lookup or default for new artists.
    # The `artist_genres_cleaned` used in prediction needs to be consistent with how it was encoded.
    # For now, it will be the most common genre for the artist as identified in get_artist_historical_metrics.
    new_song_data = pd.DataFrame([[
        days_since_release,
        release_year,
        release_month,
        release_day_of_week,
        duration_ms,
        explicit_numeric,
        available_markets_count,
        artist_followers,
        artist_popularity_snapshot,
        prev_avg_popularity,
        num_prev_releases,
        album_type,
        artist_genre_cleaned, # This should be the 'cleaned' genre string
        artist_name
    ]], columns=model_features)

    # Make prediction and calculate confidence interval
    regressor = model_pipeline.named_steps['regressor']
    preprocessor = model_pipeline.named_steps['preprocessor']

    X_processed = preprocessor.transform(new_song_data)

    individual_tree_predictions = []
    # Check if the regressor actually has 'estimators_' (e.g., RandomForest)
    if hasattr(regressor, 'estimators_'):
        for tree in regressor.estimators_:
            individual_tree_predictions.append(tree.predict(X_processed)[0])
    else:
        # Fallback if regressor is not an ensemble (e.g., Linear Regression)
        individual_tree_predictions = [regressor.predict(X_processed)[0]]


    predicted_popularity = np.mean(individual_tree_predictions)
    predicted_popularity = max(0, min(100, predicted_popularity))

    # Calculate 95% prediction interval only if we have multiple trees
    if len(individual_tree_predictions) > 1:
        lower_bound = np.percentile(individual_tree_predictions, 2.5)
        upper_bound = np.percentile(individual_tree_predictions, 97.5)
    else: # If only one prediction, range is just the prediction itself
        lower_bound = predicted_popularity
        upper_bound = predicted_popularity

    lower_bound = max(0, min(100, lower_bound))
    upper_bound = max(0, min(100, upper_bound))

    # Get Feature Importances
    feature_importances_df = pd.DataFrame()
    if hasattr(regressor, 'feature_importances_') and preprocessor is not None:
        feature_importances = regressor.feature_importances_

        # Get feature names from preprocessor (after one-hot encoding)
        # Note: If there are numerical features that weren't scaled, their names need to be explicitly included.
        # The `get_feature_names_out()` method on OneHotEncoder handles categorical features
        # The ColumnTransformer itself doesn't directly expose all final feature names,
        # so we need to reconstruct them carefully.
        # This is robust for our specific `ColumnTransformer` setup.
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(
            [col for col in model_features if col in ['album_type', 'artist_genres_cleaned', 'release_day_of_week', 'release_month', 'artist_name']]
        )
        numerical_feature_names_scaled = [col for col in model_features if col in ['days_since_release', 'release_year', 'duration_ms',
                                                                                 'explicit_numeric', 'available_markets_count',
                                                                                 'artist_followers', 'artist_popularity_snapshot',
                                                                                 'prev_avg_popularity', 'num_prev_releases']]
        all_feature_names = numerical_feature_names_scaled + list(ohe_feature_names)

        if len(feature_importances) == len(all_feature_names):
            feature_importances_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})
            feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        else:
            st.warning("Could not match feature importances to names. Displaying raw importances if available.")
            feature_importances_df = pd.DataFrame({'Importance': feature_importances})


    # Classify Success Level
    success_level = classify_popularity_level(predicted_popularity)

    return {
        'predicted_popularity': predicted_popularity,
        'popularity_lower_bound': lower_bound,
        'popularity_upper_bound': upper_bound,
        'success_level': success_level,
        'feature_importances': feature_importances_df
    }


# --- 4. Streamlit UI Layout ---

st.title("🎶 Song Popularity Prediction Dashboard")
st.markdown("""
Welcome to the Song Popularity Predictor! This dashboard uses a machine learning model
trained on historical Spotify data for artists like Gwamz and KiLLOWEN to estimate
how well a new song might perform (based on Spotify's 'popularity' score).

**Note:** The Spotify API provides a snapshot of popularity, not historical stream counts.
Predictions reflect initial impact and overall potential, not precise future stream growth over time.
""")

st.sidebar.header("Dashboard Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ("Predict New Song", "Artist Historical Trends")
)


if page_selection == "Predict New Song":
    st.header("🔮 Predict New Song Popularity")
    st.markdown("Enter details for a hypothetical new song to get a popularity prediction.")

    col1, col2 = st.columns(2)

    with col1:
        selected_artist = st.selectbox(
            "Select Artist",
            options=all_artist_names,
            help="Select the artist for whom you want to predict a new song's popularity."
        )
        new_song_release_date = st.date_input(
            "Release Date",
            datetime.now().date(),
            min_value=datetime.now().date(), # Can only predict for future/today
            help="Expected release date of the new song."
        )
        new_song_duration_ms = st.slider(
            "Duration (milliseconds)",
            min_value=60000, max_value=600000, value=200000, step=1000,
            help="Length of the track in milliseconds (e.g., 200000 ms = 3 min 20 sec)."
        )

    with col2:
        new_song_album_type = st.radio(
            "Album Type",
            ('single', 'album'),
            horizontal=True,
            help="Is this a standalone single or part of an album?"
        )
        new_song_explicit = st.checkbox(
            "Explicit Content",
            value=False,
            help="Check if the song contains explicit lyrics."
        )

    if st.button("Predict Popularity"):
        if selected_artist and new_song_release_date and new_song_duration_ms:
            with st.spinner("Calculating prediction..."):
                prediction_results = predict_song_popularity_enhanced(
                    artist_name=selected_artist,
                    album_type=new_song_album_type,
                    release_date_str=new_song_release_date.strftime('%Y-%m-%d'),
                    duration_ms=new_song_duration_ms,
                    explicit=new_song_explicit,
                    historical_df=historical_df, # Pass historical_df to helper
                    model_pipeline=model_pipeline, # Pass model pipeline
                    model_features=model_features # Pass model features
                )

                st.subheader("Prediction Results:")
                st.metric(label="Predicted Popularity Score (0-100)",
                          value=f"{prediction_results['predicted_popularity']:.2f}")
                st.info(f"**Success Level:** {prediction_results['success_level']}")
                st.write(f"Confidence Interval (95%): `{prediction_results['popularity_lower_bound']:.2f}` to `{prediction_results['popularity_upper_bound']:.2f}`")

                # Comparison to artist historical average
                artist_metrics_for_display = get_artist_historical_metrics(selected_artist, historical_df)
                if artist_metrics_for_display:
                    st.write(f"---")
                    st.write(f"**Comparison:**")
                    st.write(f"  - This prediction: `{prediction_results['predicted_popularity']:.2f}`")
                    st.write(f"  - Average for {selected_artist}: `{artist_metrics_for_display['avg_popularity']:.2f}` (based on {artist_metrics_for_display['num_releases']} past releases)")
                    st.write(f"  - Overall average (all artists): `{overall_avg_popularity:.2f}`")

                st.subheader("Key Factors (Feature Importance):")
                if not prediction_results['feature_importances'].empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=prediction_results['feature_importances'].head(10), ax=ax, palette='viridis')
                    ax.set_title("Top 10 Feature Importances for This Prediction")
                    ax.set_xlabel("Relative Importance")
                    ax.set_ylabel("Feature")
                    st.pyplot(fig)
                else:
                    st.write("Feature importances could not be calculated or displayed.")

        else:
            st.warning("Please fill in all the details for the new song.")


elif page_selection == "Artist Historical Trends":
    st.header("📈 Artist Historical Performance & Trends")
    st.markdown("Explore the past song popularity of specific artists.")

    selected_artist_for_trends = st.selectbox(
        "Select Artist to View Trends",
        options=all_artist_names,
        key="trend_artist_select"
    )

    if selected_artist_for_trends:
        artist_data_for_trends = historical_df[historical_df['artist_name'] == selected_artist_for_trends].sort_values('release_date')

        if not artist_data_for_trends.empty:
            st.subheader(f"Popularity Trends for {selected_artist_for_trends}")

            # Basic metrics for the artist
            metrics = get_artist_historical_metrics(selected_artist_for_trends, historical_df)
            if metrics:
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Releases", metrics['num_releases'])
                col_m2.metric("Avg Song Popularity", f"{metrics['avg_popularity']:.2f}")
                col_m3.metric("Artist Followers (Snapshot)", f"{metrics['artist_followers']:,}")

            # Plotting historical popularity over time
            fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
            sns.lineplot(x='release_date', y='track_popularity', data=artist_data_for_trends, marker='o', ax=ax_trend, color='teal')
            ax_trend.set_title(f"{selected_artist_for_trends}'s Song Popularity Over Time")
            ax_trend.set_xlabel("Release Date")
            ax_trend.set_ylabel("Spotify Popularity (0-100)")
            ax_trend.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_trend)

            st.markdown("---")
            st.subheader(f"Raw Data for {selected_artist_for_trends}'s Songs")
            st.dataframe(artist_data_for_trends[['track_name', 'album_type', 'release_date', 'track_popularity', 'explicit', 'duration_ms', 'available_markets_count']].set_index('track_name'))

            # NEW: Add a download button for the historical data
            csv_data = artist_data_for_trends.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv_data,
                file_name=f"{selected_artist_for_trends}_spotify_data.csv",
                mime="text/csv",
                help="Click to download the historical song data for this artist."
            )

        else:
            st.write(f"No historical data available for {selected_artist_for_trends}.")
