# Audio Emotion Classification Project

This project implements an end-to-end workflow for classifying the emotional tone of audio recordings, particularly focusing on customer service calls. It encompasses data loading, preprocessing, feature extraction, model training, and deployment with a feedback loop for continuous improvement.

## Project Purpose

The primary goal of this project is to build a system that can automatically identify the emotional state (e.g., Satisfied, Unsatisfied, Average) expressed in audio recordings of phone calls. This can be valuable for various applications, such as monitoring customer satisfaction, analyzing call center performance, or improving automated systems.

## Project Steps

1.  **Setup and Data Extraction:**
    *   **Purpose:** To prepare the environment and access the raw audio data.
    *   **Details:** This step installs required Python libraries like `librosa`, `soundfile`, `numpy`, `scikit-learn`, and `pandas`. It then extracts audio files from provided zip archives (`satisfied_calls.zip`, `unsatisfied_calls.zip`, `average_calls.zip`) into a base directory (`/content/dataset`). It also includes code to convert `.ulaw.wav` files to standard `.wav` format using `pydub` and inspect the converted files using `librosa` to verify their sample rate and duration.

2.  **Data Preparation:**
    *   **Purpose:** To create a structured mapping between audio filenames and their corresponding emotional statuses.
    *   **Details:** This step generates a CSV file (`/content/CallRecords.csv`) that serves as a central index for the audio data. It populates this CSV by scanning specific folders (`average_call_quailty_wav`, `satisfied_call_quality_wav`, `not-satisfied_call_quailty_wav`) and assigning a status based on the folder name. It also processes files from an `audio_data` folder, attempting to extract the status (avg, sat, unsat) directly from the filename using regular expressions. The script then standardizes the filenames and status entries in the CSV and creates a Python dictionary (`labels`) mapping relative file paths to their statuses.

3.  **Feature Extraction and Segmentation:**
    *   **Purpose:** To prepare the audio data for machine learning by extracting relevant features and handling longer audio files.
    *   **Details:** This step focuses on processing the audio files to extract numerical features that represent the audio content. For audio files longer than 45 seconds, it segments them into a 20-second "start" segment and a 25-second "end" segment. It then calculates standard audio features for each segment, including Mel-Frequency Cepstral Coefficients (MFCCs), Chroma features, and Mel Spectrogram features. These features are concatenated into a single feature vector for each segment. The extracted features and their corresponding labels are saved as NumPy arrays (`feature_data.npy`, `feature_labels.npy`) and a pickle file (`features.pkl`). Information about the segmentation process is also saved in `segments_info.pkl`.

4.  **Model Training:**
    *   **Purpose:** To train a machine learning model that can classify the emotional status based on the extracted audio features.
    *   **Details:** This step uses a Multilayer Perceptron (MLPClassifier) from `scikit-learn` to build a neural network model. The extracted features are first normalized using `StandardScaler`. The data is then split into training and testing sets. The MLP model is trained on the training data. The model's performance is evaluated using accuracy on the test set. The trained model and the scaler are saved using `joblib` for later use (`emotion_model.joblib`, `scaler.joblib`).

5.  **Google Drive Integration and Data Ingestion:**
    *   **Purpose:** To enable the ingestion of new audio data from external sources (specifically, Google Forms) into the processing pipeline.
    *   **Details:** This step mounts the user's Google Drive to access files. It creates a local directory (`/content/audio_data`) to store new audio files. It connects to a specified Google Sheet (presumably linked to a Google Form) to read responses, specifically the audio file URLs and the user-provided filenames and statuses. It then iterates through the form responses, matches the user-provided filename with files in a designated Google Drive folder (where form file uploads are stored), and copies the matched audio files to the local `/content/audio_data` folder. The copied files are renamed to include a suffix indicating the user-provided status (e.g., `_avg.wav`, `_sat.wav`, `_unsat.wav`).

6.  **Prediction and Retraining:**
    *   **Purpose:** To use the trained model to predict the emotion of newly ingested audio files and to incorporate these new files with their provided labels into the training data for potential model retraining.
    *   **Details:** This is the final step in the workflow. It loads the previously trained model and scaler. It then iterates through the audio files in the `/content/audio_data` folder. For each file, it extracts features from the first 20 seconds and the last 25 seconds (if the file is long enough). It uses the trained model to predict the emotional status for both the start and end segments. The script compares the predicted emotion for the end segment with the expected emotion derived from the filename suffix (which came from the Google Form response). If a mismatch is detected for files that meet the duration criteria, the features and expected label of the end segment are added to the existing feature dataset. After processing all new files, if new data was added, the model is retrained on the combined (original + new) dataset. The updated feature data, labels, and the potentially retrained model and scaler are saved.

## How to Use

1.  **Run the Notebook:** Execute all the code cells in the notebook sequentially from top to bottom. This will set up the environment, process the initial dataset, train the model, and prepare for ingesting new data.
2.  **Provide New Data:** If you have new audio recordings to classify, upload them via the Google Form linked to the specified Google Sheet. Ensure you provide the exact filename and select the correct emotional status in the form.
3.  **Re-run Step 5 and 6:** After submitting new data via the form, re-execute the code cells for Step 5 ("Google Drive Integration and Data Ingestion") and Step 6 ("Prediction and Retraining"). Step 5 will download the new audio files, and Step 6 will process them, provide predictions, and potentially retrain the model with the new data.
4.  **Check Predictions:** The output of the Step 6 code cell will display the predicted emotions for the start and end segments of each processed audio file. It will also indicate if a mismatch was detected and if retraining occurred.

## Files Generated

*   `/content/CallRecords.csv`: A CSV file containing the mapping between audio filenames and their assigned call statuses based on the initial dataset and processing.
*   `/content/segments_info.pkl`: A pickle file storing a list of dictionaries, where each dictionary contains information about a processed audio segment (original file, status, segment type, path, sample rate, original duration).
*   `/content/feature_data.npy`: A NumPy array containing the numerical feature vectors extracted from the audio segments of the initial dataset.
*   `/content/feature_labels.npy`: A NumPy array containing the corresponding emotional labels for the feature vectors in `feature_data.npy`.
*   `/content/features.pkl`: A pickle file containing a dictionary with keys 'data' (the feature data NumPy array) and 'labels' (the feature labels NumPy array).
*   `/content/emotion_model.joblib`: The trained MLPClassifier model saved in joblib format after the initial training.
*   `/content/scaler.joblib`: The StandardScaler object fitted to the initial feature data, saved in joblib format. Used for normalizing features before prediction.
*   `/content/feature_data_updated.npy`: (Generated after retraining) A NumPy array containing the combined feature data from the initial dataset and any new data used for retraining.
*   `/content/feature_labels_updated.npy`: (Generated after retraining) A NumPy array containing the combined labels from the initial dataset and any new data used for retraining.
*   `/content/emotion_model_epic_retrained.joblib`: (Generated after retraining) The retrained MLPClassifier model saved in joblib format. This model incorporates the new data from the feedback loop.
*   `/content/scaler_epic_retrained.joblib`: (Generated after retraining) The StandardScaler object fitted to the updated feature data.

## Requirements

All necessary Python libraries are listed and installed in the first code cell (`!pip install ...`). Ensure you have a Google account and access to Google Drive and Google Sheets for the data ingestion step.

## Data Structure

The project expects the initial raw audio data to be available in zip files that are extracted into the `/content/dataset` directory. Within `/content/dataset`, the script looks for specific subdirectories (`average_call_quailty_wav`, `satisfied_call_quality_wav`, `not-satisfied_call_quailty_wav`) and potentially an `audio_data` directory. New audio data is expected to be uploaded via a Google Form linked to a Google Sheet, with the actual audio files appearing in a designated Google Drive folder.
