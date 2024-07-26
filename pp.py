import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.impute import SimpleImputer

# Function to run selected algorithms
def run_algorithms(X_train, X_test, y_train, y_test, algorithms, metrics):
    results = {}
    for algorithm in algorithms:
        if algorithm == "Random Forest Classifier":
            model = RandomForestClassifier()
        elif algorithm == "Logistic Regression":
            model = LogisticRegression()
        elif algorithm == "Gradient Boosting Classifier":
            model = GradientBoostingClassifier()
        elif algorithm == "Linear Regression":
                model = LinearRegression()
        elif algorithm == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        elif algorithm == "Random Forest Regressor":
            model = RandomForestRegressor()
        elif algorithm == "Lasso Regression":
            model = Lasso()
        elif algorithm == "Support Vector Machine Classifier":
            model = SVC()
        elif algorithm == "Support Vector Machine Regressor":
            model = SVR()
        elif algorithm == "Decision Tree Regressor":
            model = DecisionTreeRegressor()  
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on training set
        train_predictions = model.predict(X_train)
        #train_accuracy = accuracy_score(y_train, train_predictions) if "Classifier" in algorithm else r2_score(y_train, train_predictions)
        # Calculate selected metrics on training set
        train_scores = calculate_metrics(y_train, train_predictions, metrics)
    
        # Make predictions on test set
        test_predictions = model.predict(X_test)
        test_scores = calculate_metrics(y_test, test_predictions, metrics)
    
        # Store scores
        results[algorithm] = {
            "Training Scores": train_scores,
            "Test Scores": test_scores
        }

    
    return results
# Function to calculate selected evaluation metrics
def calculate_metrics(y_true, y_pred, metrics):
    scores = {}
    if "Accuracy" in metrics:
        scores["Accuracy"] = accuracy_score(y_true, y_pred)
    if "Precision" in metrics:
        scores["Precision"] = precision_score(y_true, y_pred, average='weighted')
    if "Recall" in metrics:
        scores["Recall"] = recall_score(y_true, y_pred, average='weighted')
    if "F1 Score" in metrics:
        scores["F1 Score"] = f1_score(y_true, y_pred, average='weighted')
    if "R^2" in metrics:
        scores["R^2"] = r2_score(y_true, y_pred)
    if "Mean Absolute Error" in metrics:
        scores["Mean Absolute Error"] = mean_absolute_error(y_true, y_pred)
    if "Mean Squared Error" in metrics:
        scores["Mean Squared Error"] = mean_squared_error(y_true, y_pred)
    if "Root Mean Squared Error" in metrics:
        scores["Root Mean Squared Error"] = mean_squared_error(y_true, y_pred, squared=False)
    return scores

# UI
st.set_page_config(page_title="ML Algorithm Selector", page_icon=":robot:")

st.title("Automated ML Models 🧠")
st.markdown("---")
st.subheader("Upload your data and select algorithms to see results")
st.markdown("---")

# Apply custom CSS styles
st.write(
    f"""
    <style>
    body {{
        background-color: #f0f2f6;
        color: #333333;
    }}
    .stButton {{
        color: #333333;
        background-color: transparent;
        border: 2px solid #333333;
        border-radius: 5px;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }}
    .stButton:hover {{
        background-color: #333333;
        color: #ffffff;
    }}
    .stCheckbox label {{
        color: #333333;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Upload data
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Display uploaded data
    st.write("### 📊Uploaded data:")
    st.write(data)

    # Select target variable
    st.sidebar.subheader("Select Options")
    target_variable = st.sidebar.selectbox("🎯Select target variable:", data.columns)

    # Select encoding method for target variable
    encode_target = st.sidebar.checkbox("Perform Label Encoding on Target Variable")

    # Select feature variables
    feature_variables = st.sidebar.multiselect("📊Select feature variables:", data.columns)

    # Select encoding method for categorical variables
    encoding_method = st.sidebar.radio("🔤Select Encoding Method for Categorical Variables:", ("Label Encoding", "One-Hot Encoding","No Encoding"))

    # Select handling missing values method
    missing_values_handling = st.sidebar.radio("🔍Select Handling Missing Values Method:", ("No Change","Drop Rows with Missing Values", 
    "Fill Missing Values with Mean", "Fill Missing Values with Median", "Fill Missing Values with Mode"))

    # Select evaluation metrics
    selected_metrics = st.sidebar.multiselect("Select Evaluation Metrics:", [
        "Accuracy", "Precision", "Recall", "F1 Score", "R^2", 
        "Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error"
    ])

    """---"""


    # Select algorithms
    selected_algorithms = st.multiselect( "Select machine learning algorithms to apply:",
        ["Decision Tree Classifier","Decision Tree Regressor","Gradient Boosting Classifier",
         "Lasso Regression","Linear Regression","Logistic Regression","Random Forest Classifier",
         "Random Forest Regressor", "Support Vector Machine Classifier", "Support Vector Machine Regressor"]
        )

   

    if st.button("🚀Run"):

        #Handle missing values
        if missing_values_handling == "Drop Rows with Missing Values":
            data.dropna(inplace=True)
        elif missing_values_handling == "Fill Missing Values with Mean":
            imputer = SimpleImputer(strategy='mean')
            data[feature_variables] = imputer.fit_transform(data[feature_variables])
        elif missing_values_handling == "Fill Missing Values with Median":
            imputer = SimpleImputer(strategy='median')
            data[feature_variables] = imputer.fit_transform(data[feature_variables])
        elif missing_values_handling == "Fill Missing Values with Mode":
            imputer = SimpleImputer(strategy='most_frequent')
            data[feature_variables] = imputer.fit_transform(data[feature_variables])
        elif missing_values_handling== "No Change":
            pass

        # Separate columns for label encoding and one-hot encoding
        label_encode_columns = []
        one_hot_encode_columns = []
        for column in feature_variables:
            if data[column].dtype == "object":
                if len(data[column].unique()) <= 10:  # Threshold for one-hot encoding
                    one_hot_encode_columns.append(column)
                else:
                    label_encode_columns.append(column)


        # Convert categorical variables
        if encoding_method == "Label Encoding":
            label_encoder = LabelEncoder()
            for column in feature_variables:
                data[column] = label_encoder.fit_transform(data[column])
        if encoding_method == "One-Hot Encoding":
            data = pd.get_dummies(data, columns=one_hot_encode_columns, drop_first=True)
        if encoding_method=="No Encoding":
            pass    

        # Label encoding of target variable
        if encode_target:
            label_encoder = LabelEncoder()
            data[target_variable] = label_encoder.fit_transform(data[target_variable])   
        

        # Split data into features and target
        X = data[feature_variables]
        y = data[target_variable]           
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

         

        # Run selected algorithms
        results = run_algorithms(X_train, X_test, y_train, y_test, selected_algorithms,selected_metrics )
        
        # Display results
        st.write("### Results:")
        for algorithm, scores in results.items():
            st.write(f"**{algorithm}**:")
            df = pd.DataFrame(scores).T
            st.table(df)
          #  for key, value in scores.items():
          #      st.write(f"**{key}:** {value}")
    

        st.balloons()