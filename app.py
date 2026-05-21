import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import requests
from io import BytesIO

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("imdb-movies-dataset-5000.csv")

df = load_data()

# -----------------------------
# FIX NUMERICAL COLUMNS
# -----------------------------

# Votes column fix
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')

df['Votes'] = pd.to_numeric(
    df['Votes'],
    errors='coerce'
)

# Year
df['Year'] = pd.to_numeric(
    df['Year'],
    errors='coerce'
)

# Duration
df['Duration (min)'] = pd.to_numeric(
    df['Duration (min)'],
    errors='coerce'
)

# Rating
df['Rating'] = pd.to_numeric(
    df['Rating'],
    errors='coerce'
)

# Metascore
df['Metascore'] = pd.to_numeric(
    df['Metascore'],
    errors='coerce'
)

# Review Count
df['Review Count'] = pd.to_numeric(
    df['Review Count'],
    errors='coerce'
)

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------

df['Year'] = df['Year'].fillna(
    df['Year'].median()
)

df['Duration (min)'] = df['Duration (min)'].fillna(
    df['Duration (min)'].median()
)

df['Metascore'] = df['Metascore'].fillna(
    df['Metascore'].median()
)

df['Votes'] = df['Votes'].fillna(
    df['Votes'].median()
)

df['Review Count'] = df['Review Count'].fillna(
    df['Review Count'].median()
)

# Text columns
df['Certificate'] = df['Certificate'].fillna('Unknown')

df['Genre'] = df['Genre'].fillna('Unknown')

df['Director'] = df['Director'].fillna('Unknown')

df['Cast'] = df['Cast'].fillna('Unknown')

df['Review Title'] = df['Review Title'].fillna('No Title')

# Remove important missing rows
df = df.dropna(subset=['Review', 'Rating'])

# -----------------------------
# CREATE SENTIMENT COLUMN
# -----------------------------
df['Sentiment'] = df['Rating'].apply(
    lambda x: 'Positive' if x >= 7 else 'Negative'
)

# -----------------------------
# LOAD MODEL + VECTORIZER
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

cv = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🎬 IMDb Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 EDA",
        "🧠 Prediction"
    ]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if menu == "🏠 Home":

    st.title("🎬 IMDb Movie Review Sentiment Analysis")

    st.markdown(
        "### Naive Bayes NLP Classification Project"
    )

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg",
        width=250
    )

    st.write("""
    This application predicts whether a movie review is:

    - 👍 Positive
    - 👎 Negative

    using Machine Learning and NLP.
    """)

    st.info(
        "Model Used: Multinomial Naive Bayes"
    )

# -----------------------------
# DATASET PAGE
# -----------------------------
elif menu == "📊 Dataset":

    st.title("📊 Dataset Overview")

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))

    st.subheader("Columns")
    st.write(df.columns)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# -----------------------------
# EDA PAGE
# -----------------------------
elif menu == "📈 EDA":

    st.title("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    # Sentiment Distribution
    with col1:

        st.subheader("Sentiment Distribution")

        fig1, ax1 = plt.subplots()

        df['Sentiment'].value_counts().plot(
            kind='bar',
            ax=ax1
        )

        st.pyplot(fig1)

    # Ratings
    with col2:

        st.subheader("Ratings Distribution")

        fig2, ax2 = plt.subplots()

        df['Rating'].hist(
            bins=20,
            ax=ax2
        )

        st.pyplot(fig2)

    st.divider()

    col3, col4 = st.columns(2)

    # Genres
    with col3:

        st.subheader("Top Genres")

        fig3, ax3 = plt.subplots()

        df['Genre'].value_counts().head(10).plot(
            kind='bar',
            ax=ax3
        )

        st.pyplot(fig3)

    # Directors
    with col4:

        st.subheader("Top Directors")

        fig4, ax4 = plt.subplots()

        df['Director'].value_counts().head(10).plot(
            kind='bar',
            ax=ax4
        )

        st.pyplot(fig4)

    st.divider()

    # Votes Distribution
    st.subheader("Votes Distribution")

    fig5, ax5 = plt.subplots()

    df['Votes'].hist(
        bins=30,
        ax=ax5
    )

    st.pyplot(fig5)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif menu == "🧠 Prediction":

    st.title("🧠 Predict Movie Review Sentiment")

    review = st.text_area(
        "Enter Movie Review"
    )

    movie_name = st.text_input(
        "Enter Movie Title (Optional)"
    )

    if st.button("Predict Sentiment"):

        if review.strip() == "":

            st.warning(
                "Please enter a review."
            )

        else:

            review_vector = cv.transform([review])

            prediction = model.predict(
                review_vector
            )[0]

            if prediction == "Positive":

                st.success(
                    "👍 Positive Review"
                )

            else:

                st.error(
                    "👎 Negative Review"
                )

    # Poster Display
    if movie_name:

        st.subheader("🎬 Movie Poster")

        try:

            movie_row = df[
                df['Title'].str.contains(
                    movie_name,
                    case=False,
                    na=False
                )
            ]

            if not movie_row.empty:

                poster_url = movie_row.iloc[0]['Poster']

                response = requests.get(
                    poster_url
                )

                image = Image.open(
                    BytesIO(response.content)
                )

                st.image(
                    image,
                    caption=movie_name,
                    use_container_width=True
                )

            else:

                st.info(
                    "Movie not found."
                )

        except:

            st.warning(
                "Poster could not be loaded."
            )