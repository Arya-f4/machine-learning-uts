import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Set plot style for visualizations
sns.set(style='whitegrid')

def load_data(filepath):
    """
    Loads the dataset from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None

def exploratory_data_analysis(df):
    """
    Performs exploratory data analysis (EDA) and generates visualizations.

    Args:
        df (pandas.DataFrame): The input dataframe.
    """
    print("\n--- Exploratory Data Analysis ---")
    print("\nDataset Info:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    # --- Visualizations ---
    plt.figure(figsize=(12, 8))

    # Survival Count
    plt.subplot(2, 2, 1)
    sns.countplot(x='Survived', data=df)
    plt.title('Survival Count (0 = No, 1 = Yes)')

    # Survival by Sex
    plt.subplot(2, 2, 2)
    sns.countplot(x='Survived', hue='Sex', data=df)
    plt.title('Survival by Sex')

    # Survival by Pclass
    plt.subplot(2, 2, 3)
    sns.countplot(x='Survived', hue='Pclass', data=df)
    plt.title('Survival by Passenger Class')

    # Age Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(df['Age'].dropna(), kde=True, bins=30)
    plt.title('Age Distribution')

    plt.tight_layout()
    plt.savefig('eda_plots.png')
    print("\nEDA plots saved to 'eda_plots.png'")
    plt.close()


def preprocess_data(df):
    """
    Preprocesses the data by handling missing values, creating new features,
    and encoding categorical variables.

    Args:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: The preprocessed dataframe ready for modeling.
    """
    print("\n--- Data Preprocessing ---")
    df_processed = df.copy()

    # 1. Handle Missing Values
    # Fill missing 'Age' with the median age
    median_age = df_processed['Age'].median()
    df_processed['Age'].fillna(median_age, inplace=True)
    print(f"Filled missing 'Age' values with median: {median_age}")

    # Fill missing 'Embarked' with the mode
    mode_embarked = df_processed['Embarked'].mode()[0]
    df_processed['Embarked'].fillna(mode_embarked, inplace=True)
    print(f"Filled missing 'Embarked' values with mode: {mode_embarked}")

    # 2. Feature Engineering
    # Create 'FamilySize' feature
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    print("Created 'FamilySize' feature.")

    # Create 'IsAlone' feature
    df_processed['IsAlone'] = 0
    df_processed.loc[df_processed['FamilySize'] == 1, 'IsAlone'] = 1
    print("Created 'IsAlone' feature.")

    # 3. Drop Unnecessary Columns
    # 'Cabin' has too many missing values. 'Ticket', 'Name', and 'PassengerId' are not useful for this model.
    cols_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId', 'SibSp', 'Parch']
    df_processed.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped columns: {cols_to_drop}")

    # 4. Encode Categorical Variables
    # Convert 'Sex' and 'Embarked' into dummy variables
    df_processed = pd.get_dummies(df_processed, columns=['Sex', 'Embarked'], drop_first=True)
    print("Encoded 'Sex' and 'Embarked' columns.")

    print("\nPreprocessing complete. Final features:")
    print(df_processed.columns)
    return df_processed


def train_and_evaluate(df):
    """
    Trains a RandomForestClassifier and evaluates its performance.
    Handles imbalanced data using RandomOverSampler.

    Args:
        df (pandas.DataFrame): The preprocessed dataframe.
    """
    print("\n--- Model Training and Evaluation ---")

    # 1. Define Features (X) and Target (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 2. Split Data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- Handle Imbalanced Data using RandomOverSampler on the training data ---
    print("\nOriginal training set distribution:")
    print(y_train.value_counts())
    
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    print("\nResampled training set distribution (after ROS):")
    print(y_train_resampled.value_counts())
    print("Applying RandomOverSampler to balance the training data.")
    # --- End of Imbalance Handling ---

    # 3. Train the Model
    # We can also add class_weight='balanced' as an extra measure
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, class_weight='balanced')
    print("\nTraining RandomForestClassifier on resampled data...")
    model.fit(X_train_resampled, y_train_resampled) # Latih model pada data yang sudah di-resample
    print("Model training complete.")

    # 4. Make Predictions on the original test set
    y_pred = model.predict(X_test)

    # 5. Evaluate the Model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix plot saved to 'confusion_matrix.png'")
    plt.close()

def main():
    """
    Main function to run the entire pipeline.
    """
    # Path diubah untuk menunjuk ke folder induk (satu level di atas)
    filepath = '../train.csv'
    df = load_data(filepath)

    if df is not None:
        exploratory_data_analysis(df)
        df_processed = preprocess_data(df)
        train_and_evaluate(df_processed)
        print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()
