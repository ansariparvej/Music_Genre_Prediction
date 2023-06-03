import streamlit as st
import pandas as pd
import pickle


file_path = './music_genre_model.pkl'

# Load the pickled model
model = pickle.load(open(file_path, 'rb'))

# Define the loan eligibility prediction function


def music_genre(data):
    # Perform the loan eligibility prediction using the loaded model
    prediction = model.predict(data)
    return prediction


def main():
    # Set the page title
    st.title('MUSIC GENRE PREDICTION')

    # Create input fields for loan details
    tempo = st.slider('Tempo', 50, 235)
    beats = st.slider('Beats', 18, 117)
    chroma_stft = st.slider('Chroma Stft', 0.1, 0.7)
    rmse = st.slider('RMSE', 0.001, 0.40)
    spectral_centroid = st.slider('Spectral Centroid', 568, 4435)
    spectral_bandwidth = st.slider('Spectral Bandwidth', 897, 3510)
    rolloff = st.slider('Rolloff', 749, 8677)
    zero_crossing_rate = st.slider('Zero Crossing Rate', 0.01, 0.3)
    mfcc1 = st.slider('mfcc1', -553.0, 43.0)
    mfcc2 = st.slider('mfcc2', -2.0, 194.0)
    mfcc3 = st.slider('mfcc3', -88.0, 57.0)
    mfcc4 = st.slider('mfcc4', -19.0, 81.0)
    mfcc5 = st.slider('mfcc5', -39.0, 32.0)
    mfcc6 = st.slider('mfcc6', -29.0, 46.0)
    mfcc7 = st.slider('mfcc7', -33.0, 22.0)
    mfcc8 = st.slider('mfcc8', -25, 50)
    mfcc9 = st.slider('mfcc9', -32, 20)
    mfcc10 = st.slider('mfcc10', -13.0, 28.0)
    mfcc11 = st.slider('mfcc11', -29.0, 18.0)
    mfcc12 = st.slider('mfcc12', -16.0, 24.0)
    mfcc13 = st.slider('mfcc13', -28.0, 14.0)
    mfcc14 = st.slider('mfcc14', -11.0, 19.0)
    mfcc15 = st.slider('mfcc15', -18.0, 13.0)
    mfcc16 = st.slider('mfcc16', -16.0, 14.0)
    mfcc17 = st.slider('mfcc17', -18.0, 12.0)
    mfcc18 = st.slider('mfcc18', -12.0, 16.0)
    mfcc19 = st.slider('mfcc19', -19.0,15.0)
    mfcc20 = st.slider('mfcc20', -20.0, 16.0)
    

    # Prepare the loan data as input for prediction
    data = pd.DataFrame({
        'Tempo': [tempo],
        'Beats': [beats],
        'ChromaStft': [chroma_stft],
        'RMSE': [rmse],
        'SpectralCentroid': [spectral_centroid],
        'SpectralBandwidth': [spectral_bandwidth],
        'Rolloff': [rolloff],
        'ZeroCrossingRate': [zero_crossing_rate],
        'mfcc1': [mfcc1],
        'mfcc2': [mfcc2],
        'mfcc3': [mfcc3],
        'mfcc4': [mfcc4],
        'mfcc5': [mfcc5],
        'mfcc6': [mfcc6],
        'mfcc7': [mfcc7],
        'mfcc8': [mfcc8],
        'mfcc9': [mfcc9],
        'mfcc10': [mfcc10],
        'mfcc11': [mfcc11],
        'mfcc12': [mfcc12],
        'mfcc13': [mfcc13],
        'mfcc14': [mfcc14],
        'mfcc15': [mfcc15],
        'mfcc16': [mfcc16],
        'mfcc17': [mfcc17],
        'mfcc18': [mfcc18],
        'mfcc19': [mfcc19],
        'mfcc20': [mfcc20]
               
         })

    # Form submission
    if st.button('Predict'):
        # Perform Online Shoppers Intention
        prediction = music_genre(data)
        st.write("Music Genre: " , prediction[0])


if __name__ == '__main__':
    main()
