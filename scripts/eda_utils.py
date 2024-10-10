import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summary_statistics(df):
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    # Calculate basic summary statistics
    summary_stats = numeric_df.describe().T
    summary_stats['median'] = numeric_df.median()
    summary_stats['mode'] = numeric_df.mode().iloc[0]
    summary_stats['skewness'] = numeric_df.skew()
    summary_stats['kurtosis'] = numeric_df.kurtosis()
    
    return summary_stats

def plot_numeric_columns(df):
  
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include='number').columns

    # Check if there are numeric columns
    if numeric_cols.empty:
        raise ValueError("No numeric columns found in the DataFrame.")

    # Calculate number of plots and create subplots
    num_plots = len(numeric_cols)
    nrows = (num_plots + 2) // 3  # Determine number of rows needed for 3 columns

    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(18, nrows * 4))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

    # Loop through numeric columns and plot
    for idx, column in enumerate(numeric_cols):
        sns.histplot(df[column], bins=30, kde=True, color='skyblue', stat='density', edgecolor='black', ax=axes[idx])
        axes[idx].axvline(df[column].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
        axes[idx].axvline(df[column].median(), color='green', linestyle='dashed', linewidth=2, label='Median')
        
        axes[idx].set_title(f'Histogram and Line Plot of {column}')
        axes[idx].set_xlabel(column)
        axes[idx].set_ylabel('Density')
        axes[idx].legend()

    # Hide any unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_categorical_columns(df):
   
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    cols_with_few_categories = [col for col in categorical_cols if df[col].nunique() <= 10]

    # Set up the grid for subplots
    num_cols = len(cols_with_few_categories)
    num_rows = (num_cols + 1) // 2  # Automatically determine the grid size
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(cols_with_few_categories):
        ax = sns.countplot(data=df, x=col, ax=axes[i], hue=col, legend=False, palette="Set1")
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frequency')

    
    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df):
    
    # Select numerical columns for correlation analysis
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Calculate the correlation matrix
    correlation_matrix = df[numerical_cols].corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, cbar_kws={"shrink": .8}, linewidths=0.5)

    # Title and labels
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.tight_layout()
    plt.show()

def detect_outliers(df):
    """Detect outliers using box plots for numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n_cols = 2
    n_rows = (len(numerical_cols) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)
    
    # Remove unused axes 
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()