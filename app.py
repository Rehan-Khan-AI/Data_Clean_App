import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats.mstats import winsorize
from io import StringIO
import os

# Title and file uploader
st.title("Interactive Data Cleaning Tool")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Initialize session state for cleaned_df
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = df.copy()

    # Display basic information about the data
    st.subheader("Data Overview")
    
    # Collect data types and non-null counts
    data_types = st.session_state.cleaned_df.dtypes
    non_null_counts = st.session_state.cleaned_df.notnull().sum()
    
    # Create a summary DataFrame
    overview_df = pd.DataFrame({
        'Column Name': data_types.index,
        'Data Type': data_types.values,
        'Non-null Count': non_null_counts.values
    })
    
    # Display the overview DataFrame
    st.dataframe(overview_df)
    
    # Display memory usage
    buffer = StringIO()
    st.session_state.cleaned_df.info(buf=buffer)
    info_text = buffer.getvalue().split("\n")[-2]
    st.write(f"Memory Usage: {info_text.strip()}")
    
    # Data preview with configurable options
    st.subheader("Data Preview")
    head_size = st.slider("Number of rows to show at the beginning (Head)", 1, 10, 5)
    tail_size = st.slider("Number of rows to show at the end (Tail)", 1, 10, 5)
    st.write(st.session_state.cleaned_df.head(head_size))
    st.write(st.session_state.cleaned_df.tail(tail_size))

    # Display data shape (number of rows and columns) with explanation
    data_shape = st.session_state.cleaned_df.shape
    st.subheader(f"Data Shape : {data_shape[0]} rows (observations) x {data_shape[1]} columns (features)")

    # Display summary statistics (descriptive statistics) with explanation
    st.subheader("Data Summary")
    st.write("This table shows some summary statistics for the numerical columns in your data. It includes:")
    st.write("- count: The number of non-null values in each column.")
    st.write("- mean: The average value.")
    st.write("- std: The standard deviation (spread of the data around the mean).")
    st.write("- min: The minimum value.")
    st.write("- 25%: The first quartile (25% of the data falls below this value).")
    st.write("- 50%: The second quartile (median, 50% of the data falls below this value).")
    st.write("- 75%: The third quartile (75% of the data falls below this value).")
    st.write("- max: The maximum value.")
    st.write(st.session_state.cleaned_df.describe(include='all'))  # Include all data types

    # Display missing value counts with explanation
    st.subheader("Missing Value Counts")
    st.write("This table shows the number of missing values in each column of your data.")
    missing_values = st.session_state.cleaned_df.isnull().sum()
    st.write(missing_values)

    # User selection for dropping rows with missing values
    st.subheader("Drop Rows with Missing Values (Optional)")
    if st.checkbox("Enable Dropping Rows with Missing Values"):
        # Use multiselect to allow users to choose multiple columns
        columns_to_drop_missing = st.multiselect("Select columns to drop rows with missing values:", st.session_state.cleaned_df.columns)
        if columns_to_drop_missing:
            st.session_state.cleaned_df = st.session_state.cleaned_df.dropna(subset=columns_to_drop_missing)
            st.write("Preview of cleaned data (rows with missing values dropped in selected columns):")
            st.write(st.session_state.cleaned_df.head())  # Show a preview of the cleaned data

            # Check for remaining null values after dropping
            null_sum_after_drop = st.session_state.cleaned_df.isnull().sum().sum()
            if null_sum_after_drop == 0:
                st.success("**No missing values found after dropping rows!**")
                # Only display cleaned data when there are truly no missing values
                st.write("**Here's the entire cleaned data after dropping rows with missing values:**")
                st.write(st.session_state.cleaned_df)
            else:
                st.warning(f"There are still {null_sum_after_drop} missing values in the data. Missing values might exist in these columns:")
                columns_with_missing = st.session_state.cleaned_df.columns[st.session_state.cleaned_df.isnull().any()]
                st.write(columns_with_missing.to_list())  # Display list of columns with missing values

                # Illustrative sample row with zeros for user satisfaction
                if st.checkbox("Show Sample Row with Zeros (for illustrative purposes)"):
                    # Create a sample row with zeros for illustrative purposes
                    sample_row_with_zeros = pd.Series([0] * len(st.session_state.cleaned_df.columns), index=st.session_state.cleaned_df.columns)
                    st.write("Sample row after dropping missing values (all zeros for illustrative purposes):")
                    st.write(sample_row_with_zeros)

    # Outlier handling (optional)
    handle_outliers = st.checkbox("Handle Outliers", key="handle_outliers")
    if handle_outliers:
        outlier_method = st.selectbox("Outlier Handling Method:", ("Winsorize", "Remove outliers"))
        
        # Outlier visualization with Plotly box plots
        st.subheader("Outlier Detection with Box Plots")
        for col in st.session_state.cleaned_df.select_dtypes(include=["float", "int"]).columns:
            fig = px.box(st.session_state.cleaned_df, y=col, title=f"Box Plot for {col}")
            st.plotly_chart(fig)

    # Cleaned data display (optional)
    if st.button("Clean Data"):
        cleaned_df = st.session_state.cleaned_df.copy()

        # Outlier handling
        if handle_outliers:
            if outlier_method == "Winsorize":
                # Winsorize each numeric column
                for col in cleaned_df.select_dtypes(include=["float", "int"]).columns:
                    cleaned_df[col] = winsorize(cleaned_df[col], limits=[0.05, 0.05])
                st.success("Outliers have been winsorized successfully.")
            elif outlier_method == "Remove":
                # Removing outliers using IQR method
                Q1 = cleaned_df.quantile(0.25)
                Q3 = cleaned_df.quantile(0.75)
                IQR = Q3 - Q1
                is_outlier = (cleaned_df < (Q1 - 1.5 * IQR)) | (cleaned_df > (Q3 + 1.5 * IQR))
                st.write(f"Outliers detected:\n{is_outlier.sum()}")
                cleaned_df = cleaned_df[~is_outlier.any(axis=1)]
                st.success("Outliers removed successfully.")

        st.session_state.cleaned_df = cleaned_df
        st.subheader("Cleaned Data")
        st.write(st.session_state.cleaned_df)

        # Show box plots after cleaning to verify no outliers
        st.subheader("Post-Cleaning Outlier Check with Box Plots")
        for col in st.session_state.cleaned_df.select_dtypes(include=["float", "int"]).columns:
            fig = px.box(st.session_state.cleaned_df, y=col, title=f"Box Plot for {col} (Post-Cleaning)")
            st.plotly_chart(fig)



# Define the directory for exporting files
export_dir = "exports"

# Create the export directory if it doesn't exist
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# Export cleaned data
if st.button("Export Cleaned Data to CSV"):
    cleaned_df_filename = st.text_input("Enter a filename for the CSV file:", value="cleaned_data.csv")
    cleaned_df_path = os.path.join(export_dir, cleaned_df_filename)
    st.session_state.cleaned_df.to_csv(cleaned_df_path, index=False)
    st.success(f"Cleaned data exported successfully.")
    st.write(f"Exported file path: {cleaned_df_path}")  # Debugging line

    
st.stop()
