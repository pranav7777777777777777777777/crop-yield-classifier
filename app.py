
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout='wide')
st.title("🌾 Crop Yield Classification Dashboard")

# Load dataset
df = pd.read_csv("Yield Production data.csv")
st.subheader("Raw Data")
st.dataframe(df)

if st.checkbox("Show summary"):
    st.write(df.describe())

# Visualization
st.subheader("📊 Data Visualization")
selected_col = st.selectbox("Choose a feature to plot against yield", df.columns[:-1])
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=df, x=selected_col, y=df.columns[-1], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig)

# Prepare data
X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

# Convert continuous y to classification: Bin it into 3 categories (Low, Medium, High)
y = pd.qcut(y, q=3, labels=["Low", "Medium", "High"])

# Handle missing values
X_numeric = X.select_dtypes(include=['number'])
X[X_numeric.columns] = X_numeric.fillna(X_numeric.median())
X_non_numeric = X.select_dtypes(exclude=['number'])
for col in X_non_numeric.columns:
    X[col].fillna(X[col].mode()[0], inplace=True)

# Encode categorical
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_name = st.selectbox("Choose model",
    ["Random Forest", "Decision Tree", "SVM", "Logistic Regression", "Naive Bayes", "KNN"])

if model_name == "Random Forest":
    model = RandomForestClassifier()
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_name == "SVM":
    model = SVC()
elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
elif model_name == "Naive Bayes":
    model = GaussianNB()
elif model_name == "KNN":
    model = KNeighborsClassifier()

if st.button("Train Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("✅ Model Trained Successfully")
    st.subheader("📈 Model Performance")
    st.write("### Debug Info")
    st.write("Target class distribution:")
    st.write(y.value_counts())
    st.write("Predicted class distribution:")
    st.write(pd.Series(y_pred).value_counts())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.2f}")
