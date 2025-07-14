import customtkinter as ctk
from tkinter import simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
import io
import threading
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
active_plot = {
    "canvas": None,
    "function": None
}
# ========== Data Preparation ==================== #
df_raw = pd.read_csv("D:\python pj\DA sections\melb_data.csv")
df = df_raw.copy()
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)

def remove_outliers_columns(df, columns):
    df_cleaned = df.copy()
    outlier_indices = set()
    for column in columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df_cleaned[(df_cleaned[column] < lower) | (df_cleaned[column] > upper)].index
        outlier_indices.update(outliers)
    return df_cleaned.drop(index=outlier_indices).reset_index(drop=True)

columns_to_clean = ['Price', 'Landsize', 'BuildingArea']
df = remove_outliers_columns(df, columns_to_clean)

def label_encode_string_columns(df):
    df_encoded = df.copy()
    encoders = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders

df_encoded, encoders = label_encode_string_columns(df)
x = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

model_kfold = None
scaler_kfold = None
kfold_scores = None
last_report_df = None

# ========== GUI Setup ========== #
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")
app = ctk.CTk()
app.geometry("1200x700")
app.title("🏡 ML Dashboard | Melbourne Housing")

sidebar = ctk.CTkFrame(app, width=350, fg_color="#2c2f38")
sidebar.pack(side="left", fill="y", padx=(10, 0))

main_area = ctk.CTkFrame(app, corner_radius=0)

main_area.pack(side="right", fill="both", expand=True)

header = ctk.CTkLabel(main_area, text="🏠 Melbourne Housing ML Analysis", font=("Helvetica", 20, "bold"))
header.pack(pady=10)

output_box = ctk.CTkTextbox(main_area, font=("Consolas", 13), wrap="word")
main_area.grid_rowconfigure(1, weight=1)
main_area.grid_columnconfigure(0, weight=3)  
main_area.grid_columnconfigure(1, weight=2)  
header.grid(row=0, column=0, columnspan=2, pady=10)
output_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
def clear_output():
    output_box.delete("1.0", "end")
#==============The functions=============================#
def show_summary():
    clear_output()
    buf = io.StringIO()
    df.info(buf=buf)
    output_box.insert("end", f"Data Summary\n{'-'*30}\n{buf.getvalue()}\n\nShape: {df.shape}\n\nNulls:\n{df.isna().sum()}")

def train_test_split_model():
    def worker():
        app.after(0, lambda: output_box.insert("end", "⏳ Please wait... Training using Train/Test Split \n" + "="*60 + "\n"))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        r2_split = r2_score(y_test, y_pred)
        mse_split = mean_squared_error(y_test, y_pred)
        rmse_split = np.sqrt(mse_split)
        mae_split = mean_absolute_error(y_test, y_pred)
        result = f"""
✅ Train/Test Split Complete
R2 Score  : {r2_split:.4f}
MSE       : {mse_split:.4f}
RMSE      : {rmse_split:.4f}
MAE       : {mae_split:.4f}
""" + "="*60 + "\n"
        app.after(0, lambda: output_box.insert("end", result))
    clear_output()
    threading.Thread(target=worker).start()

def train_kfold():
    def worker():
        global model_kfold, scaler_kfold, kfold_scores
        app.after(0, lambda: output_box.insert("end", "⏳ Please wait... Training using K-Fold\n" + "="*60 + "\n"))
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_score = -np.inf
        best_result = None
        i = 1
        for train_idx, test_idx in kf.split(x_scaled):
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(x_scaled[train_idx], y.iloc[train_idx])
            preds = model.predict(x_scaled[test_idx])
            r2 = r2_score(y.iloc[test_idx], preds)
            mse = mean_squared_error(y.iloc[test_idx], preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y.iloc[test_idx], preds)
            if r2 > best_score:
                best_score = r2
                best_result = (r2, mse, rmse, mae)
                model_kfold = model
            app.after(0, lambda i=i, r2=r2, mse=mse, rmse=rmse, mae=mae: output_box.insert(
                "end",
                f"✅ Iteration {i} complete\n"
                f"R2 Score : {r2:.4f}\n"
                f"MSE      : {mse:.2f}\n"
                f"RMSE     : {rmse:.2f}\n"
                f"MAE      : {mae:.2f}\n\n"
            ))
            i += 1
        scaler_kfold = scaler
        kfold_scores = best_result
        summary = f"\n✅ K-Fold Training Complete\nBest R2 Score: {best_result[0]:.4f}\n" + "="*60 + "\n"
        app.after(0, lambda: output_box.insert("end", summary))
    clear_output()
    threading.Thread(target=worker).start()


def final_report():
    def worker():
        global last_report_df
        app.after(0, lambda: output_box.insert("end", "⏳ Please wait... Generating Final Report\n" + "="*80 + "\n"))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        r2_split = r2_score(y_test, y_pred)
        mse_split = mean_squared_error(y_test, y_pred)
        rmse_split = np.sqrt(mse_split)
        mae_split = mean_absolute_error(y_test, y_pred)

        if kfold_scores is None:
            output_box.insert("end", "⚠ Please train the model with K-Fold first.\n")
            return

        r2, mse, rmse, mae = kfold_scores
        report = pd.DataFrame({
            "Metric": ["R2", "MSE", "RMSE", "MAE"],
            "Train/Test Split": [r2_split, mse_split, rmse_split, mae_split],
            "K-Fold CV": [r2, mse, rmse, mae]
        })
        
        pd.set_option('display.float_format', '{:.4f}'.format)

        last_report_df = report

        formatted_report = "\n📊 Model Evaluation Report\n" + "="*80 + "\n"
        formatted_report += report.to_string(index=False, col_space=20, justify='center')
        formatted_report += "\n" + "="*80 + "\n"
        app.after(0, lambda: output_box.insert("end", formatted_report))
    clear_output()
    threading.Thread(target=worker).start()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    r2_split = r2_score(y_test, y_pred)
    mse_split = mean_squared_error(y_test, y_pred)
    rmse_split = np.sqrt(mse_split)
    mae_split = mean_absolute_error(y_test, y_pred)

    if not kfold_scores:
        output_box.insert("end", "Train model with Kfold\n")
        return
    r2 = kfold_scores
    mse = np.mean([r[1] for r in kfold_scores])
    rmse = np.mean([r[2] for r in kfold_scores])
    mae = np.mean([r[3] for r in kfold_scores])

    report = pd.DataFrame({
        "Metric": ["R2", "MSE", "RMSE", "MAE"],
        "Train/Test Split": [r2_split, mse_split, rmse_split, mae_split],
        "K-Fold CV": [r2, mse, rmse, mae]
    })


def predict_sample():
    clear_output()
    if model_kfold is None or scaler_kfold is None:
        output_box.insert("end", "⚠ Please train the model first.\n")
        return

    bathroom = simpledialog.askstring("Input", "Please enter the number of Bathrooms:")
    bedroom2 = simpledialog.askstring("Input", "Please enter the number of Bedroom2:")
    rooms = simpledialog.askstring("Input", "Please enter the number of Rooms:")
    property_type = simpledialog.askstring("Input", "Please enter the Property Type (e.g., h for house, u for unit):")

    if not (bathroom and bedroom2 and rooms and property_type):
        output_box.insert("end", "⚠ One or more inputs are missing.\n")
        return

    sample_data = {
        'Bathroom': int(bathroom),
        'Bedroom2': int(bedroom2),
        'Rooms': int(rooms),
        'Type': property_type
    }

    sample = pd.DataFrame([sample_data])

    for col in sample.columns:
        if col in encoders:
            le = encoders[col]
            sample[col] = sample[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    for col in x.columns:
        if col not in sample:
            sample[col] = 0
    sample = sample[x.columns]

    scaled = scaler_kfold.transform(sample)
    price = model_kfold.predict(scaled)[0]
    output_box.insert("end", f"💰 Predicted Price: {price:,.0f} EGP\n")

def run_plot(plot_func):
    global active_plot
    clear_output()

    # If same plot clicked again, toggle it (off)
    if active_plot["function"] == plot_func:
        if active_plot["canvas"]:
            active_plot["canvas"].get_tk_widget().destroy()
        active_plot["canvas"] = None
        active_plot["function"] = None
        return

    # If a different plot is active, remove it
    if active_plot["canvas"]:
        active_plot["canvas"].get_tk_widget().destroy()

    # Show the new plot
    fig = plot_func()
    canvas = FigureCanvasTkAgg(fig, master=main_area)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
    active_plot["canvas"] = canvas
    active_plot["function"] = plot_func


def plot_heatmap():
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    return fig

def plot_multi_histogram_gui():
    # Ask for the numeric column
    x_col = simpledialog.askstring("Multi Histogram", "Enter the numeric column to plot (e.g., 'Price') or leave it blank for all numeric columns:")

    if not x_col:
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            output_box.insert("end", "❌ No numeric columns found in the dataset.\n")
            return None

        output_box.insert("end", "📊 Showing multi-histograms for all numeric columns.\n")

        # Create subplots for each numeric column
        num_plots = len(numeric_cols)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 5))

        if num_plots == 1:
            axes = [axes]

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], bins=50, kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        # Adjust layout to make it look nice
        plt.tight_layout()
        return fig

   
    else:  
        if x_col not in df.columns:
            output_box.insert("end", "❌ The numeric column does not exist in the dataset.\n")
            return None

        if not pd.api.types.is_numeric_dtype(df[x_col]):
            output_box.insert("end", f"❌ Column '{x_col}' is not numeric.\n")
            return None

        hue_col = simpledialog.askstring("Multi Histogram", f"Enter the categorical column for hue (e.g., 'Type') for '{x_col}', or leave blank for no hue:")

        if hue_col:  # If the hue column is provided
            if hue_col not in df.columns:
                output_box.insert("end", f"❌ The hue column '{hue_col}' does not exist in the dataset.\n")
                return None

            if not pd.api.types.is_categorical_dtype(df[hue_col]) and not pd.api.types.is_object_dtype(df[hue_col]):
                output_box.insert("end", f"❌ Column '{hue_col}' is not categorical.\n")
                return None

            output_box.insert("end", f"📊 Showing multi-histogram for '{x_col}' by '{hue_col}'.\n")
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(df, x=x_col, hue=hue_col, kde=True, multiple='stack', palette='coolwarm')
            plt.title(f'{x_col} Distribution by {hue_col}')
            return fig  # Show the plot for the selected column
        else:  # If no hue column is provided
            output_box.insert("end", f"📊 Showing histogram for '{x_col}' without hue.\n")
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(df, x=x_col, kde=True, color='skyblue')
            plt.title(f'{x_col} Distribution')
            return fig

def plot_boxplot_gui():
    # Ask for the numeric column
    col = simpledialog.askstring("Boxplot", "Enter the numeric column to visualize, or type 'all' for all numeric columns:")

    # Ask for an optional hue (categorical column)
    hue_col = simpledialog.askstring("Boxplot", "Enter a categorical column for hue (optional, press Enter to skip):")

    # If the user chose 'all', process all numeric columns
    if col.lower() == "all":
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            output_box.insert("end", "❌ No numeric columns found in the dataset.\n")
            return None

        output_box.insert("end", "📦 Showing boxplots for all numeric columns.\n")

        # Create subplots for each numeric column
        num_plots = len(numeric_cols)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 5))

        # If only one plot, axes will not be a list, so make sure it’s iterable
        if num_plots == 1:
            axes = [axes]

        # Loop through numeric columns and plot each boxplot in a subplot
        for i, col in enumerate(numeric_cols):
            if hue_col and hue_col in df.columns and df[hue_col].dtype in ['object', 'category']:
                sns.boxplot(x=df[hue_col], y=df[col], ax=axes[i], palette='coolwarm')
                axes[i].set_title(f'{col} Boxplot by {hue_col}')
                axes[i].set_xlabel(hue_col)
                axes[i].set_ylabel(col)
            else:
                sns.boxplot(x=df[col], ax=axes[i], color='lightgreen')
                axes[i].set_title(f'{col} Boxplot')
                axes[i].set_xlabel(col)

        # Adjust layout to make it look nice
        plt.tight_layout()
        return fig

    # If the user selected a specific numeric column
    if col not in df.columns:
        output_box.insert("end", f"❌ Column '{col}' does not exist in the dataset.\n")
        return None

    if not np.issubdtype(df[col].dtype, np.number):
        output_box.insert("end", f"❌ Column '{col}' is not numeric.\n")
        return None

    output_box.insert("end", f"📦 Showing boxplot for '{col}'.\n")

    # Create the boxplot for the selected column with optional hue
    fig = plt.figure(figsize=(10, 6))
    if hue_col and hue_col in df.columns and df[hue_col].dtype in ['object', 'category']:
        sns.boxplot(x=df[hue_col], y=df[col], color='lightgreen', palette='coolwarm')
        plt.title(f'{col} Boxplot by {hue_col}')
        plt.xlabel(hue_col)
        plt.ylabel(col)
    else:
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f'{col} Boxplot')
        plt.xlabel(col)
    
    return fig

def plot_pie_chart_gui():
    col = simpledialog.askstring("Pie Chart", "Enter the categorical column to visualize (Leave it blank for all columns):")

    if not col:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns  # Get all categorical columns
    
        if not categorical_cols.any():
            output_box.insert("end", "❌ No categorical columns found.\n")
            return None

        n_cols = 2  
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))  # Adjust the figsize based on rows/columns

        # Flatten the axes array in case there are multiple rows
        axes = axes.flatten()

        for idx, col in enumerate(categorical_cols):
            colatt = df[col].dropna().value_counts()

            if len(colatt) > 20:
                colatt = colatt[:20]
                output_box.insert("end", f"⚠️ Showing top 20 categories for '{col}'.\n")

            explode = [0.05] * len(colatt)

            # Plot each pie chart in its corresponding subplot
            axes[idx].pie(
                colatt,
                labels=colatt.index,
                autopct='%1.1f%%',
                explode=explode,
                shadow=True,
                textprops={'color': 'black', 'weight': 'bold', 'fontsize': 12.5}
            )
            axes[idx].set_title(f"Distribution of '{col}'", fontsize=14, weight='bold')

        # Hide any unused subplots (if there are less than 'n_rows' * 'n_cols' categories)
        for i in range(len(categorical_cols), len(axes)):
            axes[i].axis('off')  # Hide the axis for empty subplots

        plt.tight_layout()
        return fig
    
    if col not in df.columns:
        output_box.insert("end", f"❌ Column '{col}' does not exist in the dataset.\n")
        return None

    if np.issubdtype(df[col].dtype, np.number):
        output_box.insert("end", f"❌ '{col}' is numeric. Pie charts are for categories only.\n")
        return None

    colatt = df[col].dropna().value_counts()

    if len(colatt) > 20:
        colatt = colatt[:20]
        output_box.insert("end", "⚠️ Showing top 20 categories only.\n")

    explode = [0.05] * len(colatt)

    fig = plt.figure(figsize=(8, 8))
    plt.pie(
        colatt,
        labels=colatt.index,
        autopct='%1.1f%%',
        explode=explode,
        shadow=True,
        textprops={'color': 'black', 'weight': 'bold', 'fontsize': 12.5}
    )
    plt.title(f"Distribution of '{col}'", fontsize=14, weight='bold')
    plt.tight_layout()

    return fig

def plot_bar_chart_gui():
    # Ask user for a column or choose to use all categorical columns
    col = simpledialog.askstring("Bar Chart", "Enter the column to visualize (or leave empty for all categorical columns):")

    # Define max number of categories to display
    max_categories = 20

    if col == "":
        # Handle case for plotting all categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            output_box.insert("end", "❌ No categorical columns in the dataset.\n")
            return None
        
        output_box.insert("end", f"📊 Showing bar charts for all categorical columns.\n")
        
        # Create subplots for all categorical columns
        num_cols = len(categorical_cols)
        fig, axes = plt.subplots(nrows=(num_cols // 2) + (num_cols % 2), ncols=2, figsize=(10, 6))
        axes = axes.flatten()
        
        for idx, col in enumerate(categorical_cols):
            ax = axes[idx]
            value_counts = df[col].value_counts()

            # Limit categories to max_categories or group less frequent ones into "Other"
            if len(value_counts) > max_categories:
                top_categories = value_counts.head(max_categories)
                other_categories = value_counts.tail(len(value_counts) - max_categories).sum()
                top_categories = top_categories._append(pd.Series({'Other': other_categories}))
                value_counts = top_categories

            value_counts.plot.bar(ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)  # Rotate labels for better readability
        
        # Hide any unused subplots if the number of columns is odd
        for idx in range(num_cols, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    else:
        # Handle case for plotting a specific column
        if col not in df.columns:
            output_box.insert("end", f"❌ Column '{col}' does not exist in the dataset.\n")
            return None

        output_box.insert("end", f"📊 Showing bar chart for '{col}'.\n")

        fig = plt.figure(figsize=(10, 6))
        value_counts = df[col].value_counts()

        # Limit categories to max_categories or group less frequent ones into "Other"
        if len(value_counts) > max_categories:
            top_categories = value_counts.head(max_categories)
            other_categories = value_counts.tail(len(value_counts) - max_categories).sum()
            top_categories = top_categories._append(pd.Series({'Other': other_categories}))
            value_counts = top_categories

        value_counts.plot.bar(color='skyblue', edgecolor='black')
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)  # Rotate labels for better readability
        return fig

def plot_strip_plot_gui():
    col_x = simpledialog.askstring("Strip Plot", "Enter the categorical column for x-axis (e.g., 'Type'):")
    col_y = simpledialog.askstring("Strip Plot", "Enter the numeric column for y-axis (e.g., 'Price'):")

    if not col_x or not col_y:
        output_box.insert("end", "❌ Missing column input.\n")
        return None

    if col_x not in df.columns or col_y not in df.columns:
        output_box.insert("end", f"❌ Columns '{col_x}' or '{col_y}' do not exist in the dataset.\n")
        return None

    if not np.issubdtype(df[col_y].dtype, np.number):
        output_box.insert("end", f"❌ Column '{col_y}' is not numeric.\n")
        return None

    output_box.insert("end", f"📊 Showing strip plot for '{col_y}' by '{col_x}'.\n")

    fig = plt.figure(figsize=(10, 6))
    sns.stripplot(x=col_x, y=col_y, data=df, jitter=True, palette='Set1')
    plt.title(f'{col_y} Distribution by {col_x}')
    return fig

def plot_count_plot_gui():
    max_categories = 20
    max_hue_categories = 20

    col = simpledialog.askstring("Count Plot", "Enter the categorical column to visualize (or leave empty for all):")
    hue_col = simpledialog.askstring("Count Plot", "Enter optional hue column (or leave empty):")

    if hue_col and hue_col not in df.columns:
        output_box.insert("end", f"❌ Hue column '{hue_col}' does not exist in the dataset.\n")
        return None

    if hue_col and df[hue_col].nunique() > max_hue_categories:
        output_box.insert("end", f"⚠️ Hue column '{hue_col}' has too many categories. Limiting to top {max_hue_categories}.\n")
        top_hue = df[hue_col].value_counts().nlargest(max_hue_categories).index
        df_filtered = df[df[hue_col].isin(top_hue)]
    else:
        df_filtered = df.copy()

    if col == "":
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) == 0:
            output_box.insert("end", "❌ No categorical columns in the dataset.\n")
            return None

        output_box.insert("end", f"📊 Showing count plots for all categorical columns.\n")
        num_plots = len(cat_cols)
        rows = (num_plots + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
        axes = axes.flatten()

        for idx, column in enumerate(cat_cols):
            ax = axes[idx]
            value_counts = df_filtered[column].value_counts()

            if len(value_counts) > max_categories:
                top = value_counts.head(max_categories).index
                df_filtered[column] = df_filtered[column].where(df_filtered[column].isin(top), 'Other')

            sns.countplot(x=column, data=df_filtered, hue=hue_col if hue_col else None, palette='Set2', ax=ax)
            ax.set_title(f'{column} Count')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)

        for i in range(len(cat_cols), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    else:
        if col not in df.columns:
            output_box.insert("end", f"❌ Column '{col}' does not exist in the dataset.\n")
            return None

        output_box.insert("end", f"📊 Showing count plot for '{col}'.\n")

        value_counts = df_filtered[col].value_counts()
        if len(value_counts) > max_categories:
            top = value_counts.head(max_categories).index
            df_filtered[col] = df_filtered[col].where(df_filtered[col].isin(top), 'Other')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=col, data=df_filtered, hue=hue_col if hue_col else None, palette='Set2', ax=ax)
        ax.set_title(f'{col} Count')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        return fig

def plot_pair_plot_with_hue_gui():
    cols_input = simpledialog.askstring("Pair Plot with Hue", "Enter numeric columns separated by commas (leave blank for auto selection):")

    if cols_input:
        cols = [col.strip() for col in cols_input.split(",") if col.strip()]
        for col in cols:
            if col not in df.columns:
                output_box.insert("end", f"❌ Column '{col}' does not exist.\n")
                return None
            if not pd.api.types.is_numeric_dtype(df[col]):
                output_box.insert("end", f"❌ Column '{col}' is not numeric.\n")
                return None
    else:
        numeric_cols = df.select_dtypes(include='number')
        if numeric_cols.shape[1] == 0:
            output_box.insert("end", "❌ No numeric columns found.\n")
            return None

        if numeric_cols.shape[1] > 6:
            output_box.insert("end", f"⚠️ Too many numeric columns ({numeric_cols.shape[1]}). Showing top 6 with highest variance.\n")
            cols = numeric_cols.var().sort_values(ascending=False).head(6).index.tolist()
        else:
            cols = numeric_cols.columns.tolist()

    hue_col = simpledialog.askstring("Pair Plot with Hue", "Enter the categorical column for hue (optional):")
    if hue_col:
        if hue_col not in df.columns:
            output_box.insert("end", f"❌ Hue column '{hue_col}' does not exist.\n")
            return None
        if not pd.api.types.is_categorical_dtype(df[hue_col]) and not pd.api.types.is_object_dtype(df[hue_col]):
            output_box.insert("end", f"❌ Hue column '{hue_col}' is not categorical.\n")
            return None
        if df[hue_col].nunique() > 15:
            output_box.insert("end", f"⚠️ Hue column '{hue_col}' has too many categories. Grouping rare ones into 'Other'.\n")
            top_hue = df[hue_col].value_counts().nlargest(14).index
            df[hue_col] = df[hue_col].where(df[hue_col].isin(top_hue), other='Other')

    output_box.insert("end", f"📊 Showing pair plot for: {', '.join(cols)}" + (f" with hue '{hue_col}'" if hue_col else "") + ".\n")

    # Sample to improve performance
    sample_df = df[cols + ([hue_col] if hue_col else [])].dropna()
    if sample_df.shape[0] > 1000:
        sample_df = sample_df.sample(n=1000, random_state=42)

    # Faster plotting
    sns_plot = sns.pairplot(sample_df, hue=hue_col if hue_col else None, palette='Set1', diag_kind='hist')
    return sns_plot.fig

def DistributionPlotGUI():
    max_categories = 20

    # Get user input using 
    col_filter = simpledialog.askstring("Filter Column", "Enter the column you want to filter by:")
    col_value = simpledialog.askstring("Plot Column", "Enter the numeric column you want to plot the distribution for:")

    if col_filter not in df.columns or col_value not in df.columns:
        output_box.insert("end", "❌ One or both column names do not exist in the dataset.\n")
        return None

    if not pd.api.types.is_numeric_dtype(df[col_value]):
        output_box.insert("end", f"❌ Column '{col_value}' must be numeric for distribution plotting.\n")
        return None

    if df[col_filter].dtype not in ['object', 'category'] and df[col_filter].nunique() > 10:
        output_box.insert("end", "❌ The filter column should be categorical or have a small number of unique values.\n")
        return None

    unique_values = df[col_filter].dropna().unique()
    n = len(unique_values)

    if n == 0:
        output_box.insert("end", "❌ No unique values found in the filter column.\n")
        return None

    if n > max_categories:
        output_box.insert("end", f"⚠️ The filter column has {n} unique values. Limiting to top {max_categories} frequent values.\n")
        top_values = df[col_filter].value_counts().nlargest(max_categories).index
        df_filtered = df[df[col_filter].isin(top_values)]
        unique_values = top_values
        n = len(unique_values)
    else:
        df_filtered = df.copy()

    # Create subplots
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    for i, value in enumerate(unique_values):
        subset = df_filtered[df_filtered[col_filter] == value][col_value].dropna()
        sns.histplot(subset, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f"{col_value} for {col_filter} = {value}", fontsize=12, weight='bold')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Distribution of '{col_value}' grouped by '{col_filter}'", fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    return fig

def plot_kde_plot_gui():
    col = simpledialog.askstring("KDE Plot", "Enter the numeric column:")

    # Handle empty input or missing column
    if not col:
        output_box.insert("end", "❌ No column name provided.\n")
        return None

    if col not in df.columns:
        output_box.insert("end", f"❌ Column '{col}' not found in the dataset.\n")
        return None

    # Ensure the column is numeric
    if not pd.api.types.is_numeric_dtype(df[col]):
        output_box.insert("end", f"❌ Column '{col}' is not numeric.\n")
        return None

    # Drop missing values before plotting
    data_clean = df[col].dropna()
    if data_clean.empty:
        output_box.insert("end", f"❌ Column '{col}' has no valid numeric data.\n")
        return None

    # Plot KDE
    fig = plt.figure(figsize=(10, 6))
    sns.kdeplot(data_clean, shade=True, color='green')
    plt.title(f'{col} - KDE Plot')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    output_box.insert("end", f"📈 Showing KDE plot for '{col}'.\n")
    return fig

def plot_scatter_plot_gui():
    max_hue_categories = 20

    x_col = simpledialog.askstring("Scatter Plot", "Enter the X (numeric) column:")
    y_col = simpledialog.askstring("Scatter Plot", "Enter the Y (numeric) column:")
    hue_col = simpledialog.askstring("Scatter Plot", "Enter the categorical Hue column (optional):")

    if not x_col or not y_col:
        output_box.insert("end", "❌ X and Y columns are required.\n")
        return None

    # Check if columns exist
    for col in [x_col, y_col] + ([hue_col] if hue_col else []):
        if col not in df.columns:
            output_box.insert("end", f"❌ Column '{col}' does not exist.\n")
            return None

    # Check numeric types
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        output_box.insert("end", "❌ Both X and Y columns must be numeric.\n")
        return None

    df_filtered = df[[x_col, y_col]].copy()

    if hue_col:
        df_filtered[hue_col] = df[hue_col]

        if df[hue_col].nunique() > max_hue_categories:
            output_box.insert("end", f"⚠️ Hue column '{hue_col}' has too many categories. Limiting to top {max_hue_categories}.\n")
            top_hues = df[hue_col].value_counts().nlargest(max_hue_categories).index
            df_filtered = df_filtered[df_filtered[hue_col].isin(top_hues)]

    df_filtered = df_filtered.dropna()

    if df_filtered.empty:
        output_box.insert("end", "❌ No valid data to plot after removing missing values.\n")
        return None

    # Plot
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df_filtered, hue=hue_col if hue_col else None, palette='viridis')
    plt.title(f'Scatter Plot of {y_col} vs {x_col}', fontsize=14, weight='bold')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    msg = f"📍 Showing scatter plot of '{y_col}' vs '{x_col}'" + (f" colored by '{hue_col}'." if hue_col else ".")
    output_box.insert("end", msg + "\n")

    return fig

def plot_average_numeric_by_category_gui():
    max_categories = 20

    cat_col = simpledialog.askstring("Average Plot", "Enter the categorical column (e.g., Type):")
    num_col = simpledialog.askstring("Average Plot", "Enter the numeric column (e.g., Price):")

    if not cat_col or not num_col:
        output_box.insert("end", "❌ Both column names must be entered.\n")
        return None

    if cat_col not in df.columns:
        output_box.insert("end", f"❌ Categorical column '{cat_col}' does not exist.\n")
        return None

    if num_col not in df.columns:
        output_box.insert("end", f"❌ Numeric column '{num_col}' does not exist.\n")
        return None

    if not pd.api.types.is_numeric_dtype(df[num_col]):
        output_box.insert("end", f"❌ Column '{num_col}' is not numeric.\n")
        return None

    # Handle too many categories
    if df[cat_col].nunique() > max_categories:
        output_box.insert("end", f"⚠️ Column '{cat_col}' has too many categories. Showing top {max_categories}.\n")
        top_cats = df[cat_col].value_counts().nlargest(max_categories).index
        df_filtered = df[df[cat_col].isin(top_cats)]
    else:
        df_filtered = df.copy()

    df_filtered = df_filtered[[cat_col, num_col]].dropna()

    if df_filtered.empty:
        output_box.insert("end", "❌ No valid data to plot after removing missing values.\n")
        return None

    # Compute and plot
    avg_values = df_filtered.groupby(cat_col)[num_col].mean().sort_values(ascending=True)

    fig = plt.figure(figsize=(10, 6))
    ax = avg_values.plot.barh(color='salmon')
    ax.set_xlabel(f"Average {num_col}")
    ax.set_ylabel(cat_col)
    ax.set_title(f"Average {num_col} by {cat_col}")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    output_box.insert("end", f"📊 Showing average '{num_col}' by '{cat_col}'.\n")
    return fig

# ========== Sidebar Buttons =============== #
def sidebar_button(text, command):
    return ctk.CTkButton(sidebar, text=text, command=command, font=("Arial", 14), corner_radius=10, hover_color="#2e8b57", height=15)
ctk.CTkLabel(sidebar, text="📈 Visuals", font=("Arial", 16, "bold"), text_color="white").pack(pady=(20, 10))
sidebar_button("🔥 Correlation Heatmap", lambda: run_plot(plot_heatmap)).pack(pady=5, padx=10, fill="x")
sidebar_button("📊 Histogram", lambda: run_plot(plot_multi_histogram_gui)).pack(pady=5, padx=10, fill="x")
sidebar_button("📦 Boxplot", lambda: run_plot(plot_boxplot_gui)).pack(pady=5, padx=10, fill="x")
sidebar_button("🥧 Pie Chart", lambda: run_plot(plot_pie_chart_gui)).pack(pady=5, padx=10, fill="x")
sidebar_button("📊 Bar Chart", lambda: run_plot(plot_bar_chart_gui)).pack(pady=5, padx=10, fill="x")
sidebar_button("💥 Strip Plot", lambda: run_plot(plot_strip_plot_gui)).pack(pady=5, padx=10, fill="x")
sidebar_button("🔢 Count Plot", lambda: run_plot(plot_count_plot_gui)).pack(pady=5, padx=10, fill="x")
sidebar_button("🔄 Pair Plot", lambda: run_plot(plot_pair_plot_with_hue_gui)).pack(pady=5, padx=10, fill="x")
sidebar_button("📈 Distribution Plot", lambda: run_plot(DistributionPlotGUI)).pack(pady=5, padx=10, fill="x")
sidebar_button("🌊 KDE Plot", lambda: run_plot(plot_kde_plot_gui)).pack(pady=5, padx=10, fill="x")
sidebar_button("🟢 Scatter Plot", lambda: run_plot(plot_scatter_plot_gui)).pack(pady=5, padx=10, fill="x")
sidebar_button("📊 Avg by Category", lambda: run_plot(plot_average_numeric_by_category_gui)).pack(pady=5, padx=10, fill="x")
ctk.CTkLabel(sidebar, text="🔧 Options", font=("Arial", 18, "bold"), text_color="white").pack(pady=(20, 10))
sidebar_button("📄 Data Summary", show_summary).pack(pady=5, padx=10, fill="x")
sidebar_button("🏋 Train KFold", train_kfold).pack(pady=5, padx=10, fill="x")
sidebar_button("🏋 Train test split", train_test_split_model).pack(pady=5, padx=10, fill="x")
sidebar_button("🔮 Predict Sample", predict_sample).pack(pady=5, padx=10, fill="x")
sidebar_button("⚙️ Clear Output", clear_output).pack(pady=5, padx=10, fill="x")
ctk.CTkLabel(sidebar, text="📊 Report", font=("Arial", 16, "bold"), text_color="white").pack(pady=(20, 10))
sidebar_button("📊 Final Report", final_report).pack(pady=5, padx=10, fill="x")
app.mainloop()
