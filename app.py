import streamlit as st
import pickle
import numpy as np

# --- Configuration ---
MODEL_PATH = 'linear_reg_model.pkl'

# --- Load the Model ---
# The model is loaded once when the app starts, using st.cache_resource
# for better performance.
@st.cache_resource
def load_model(path):
    """Loads the pickled model from the specified path."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{path}' not found. "
                 "Please ensure it is in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the model
model = load_model(MODEL_PATH)

# --- Streamlit UI and Logic ---
def main():
    st.title("🏡 Simple Linear Regression Predictor")
    st.markdown("Enter the feature value below to get a prediction from the pre-trained model.")

    if model is None:
        return # Stop if model loading failed

    # Identify the expected number of features (assumes the model was trained with 1 feature)
    # The provided model object only shows coef_ and intercept_, suggesting a simple linear model.
    # A real-world app would need to know the exact feature names/count.
    try:
        # Check if the model has a 'n_features_in_' attribute (common in scikit-learn)
        # The pickle file shows n_features_in_ as 1 (K )
        num_features = getattr(model, 'n_features_in_', 1) 
    except AttributeError:
        # Fallback if the attribute is not available
        num_features = 1 

    if num_features == 1:
        # Simple case: only one input feature
        input_value = st.number_input("Input Feature Value:", min_value=0.0, step=0.1, value=5.0)
        
        # Prepare the input for the model (must be a 2D array: [[feature_value]])
        input_data = np.array([[input_value]])
        
    elif num_features > 1:
        st.warning(f"The model expects **{num_features}** input features. This example only supports one input field.")
        st.stop()
    else:
        st.error("Could not determine the number of required features for the model.")
        return


    if st.button("Predict"):
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]

            # Display results
            st.subheader("Prediction Result")
            st.success(f"The predicted output is: **{prediction:.2f}**")

            # Optional: Display model coefficients (for transparency)
            st.info(f"Model Intercept: {model.intercept_:.2f}")
            if num_features == 1:
                 st.info(f"Model Coefficient (Slope): {model.coef_[0]:.2f}")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
