# Experiment 1: Data Analysis

**Name:** Rudransh Sogani  
**SAP ID:** 500120440  

## Objective
To perform exploratory data analysis on a structured dataset to derive meaningful insights using Python data manipulation and visualization libraries.

## Procedure & Implementation
The experiment utilizes Pandas for data manipulation, alongside Matplotlib and Seaborn for data visualization. The dataset `employee_information_100.csv` is loaded and inspected for structural properties before executing the following analytical tasks:

1.  **Average Salary by Department:** Grouping records by department <sup>(`df.groupby()`)</sup> to compute mean salaries <sup>(`mean()`)</sup>, visualized as a horizontal bar chart <sup>(`plot(kind="barh")`)</sup>.
2.  **Employee Count per Department:** Calculating the frequency distribution <sup>(`value_counts()`)</sup> of employees across departments, represented via a count plot <sup>(`sns.countplot()`)</sup>.
3.  **Gender Distribution:** Aggregating gender frequencies and displaying the proportions using a pie chart <sup>(`plt.pie()`)</sup>.
4.  **Salary Distribution:** Plotting a histogram with Kernel Density Estimation (KDE) <sup>(`sns.histplot(kde=True)`)</sup> to observe the frequency distribution of salary figures <sup>(`Series`)</sup>.
5.  **Experience vs. Salary Relationship:** Generating a scatter plot <sup>(`sns.scatterplot()`)</sup> to identify correlations between years of experience and salary, categorized by department and gender.
6.  **Top 10 Earners:** Extracting the records of the ten employees with the highest salaries <sup>(`nlargest()`, `DataFrame`)</sup>.
7.  **Departmental Maximum Salaries:** Identifying the peak salary figure within each department <sup>(`groupby().max()`)</sup>.
8.  **Above-Average Earners:** Filtering employees whose salaries exceed the overall mean salary <sup>(`df[df['Salary'] > df['Salary'].mean()]`)</sup> of the organization.
9.  **Average Experience per Department:** Computing the mean years of experience grouped by department <sup>(`groupby().mean()`)</sup>.
10. **Age Distribution:** Visualizing the demographic age spread through a histogram with KDE <sup>(`sns.histplot()`)</sup>.

## Observations
| Analysis Task | Key Observation Parameter | Visualization/Method |
| :--- | :--- | :--- |
| **Departmental Salary** | Mean salary variance across distinct departments | Horizontal Bar Chart |
| **Employee Distribution** | Workforce concentration per department | Count Plot |
| **Gender Demographics** | Proportion of male vs female employees | Pie Chart |
| **Salary Frequency** | Salary spread and common compensation brackets | Histogram with KDE |
| **Experience & Pay** | Correlation between years worked and compensation | Scatter Plot (Hue: Dept, Style: Gender) |
| **Top Earners** | Identification of 10 highest-paid employees | Dataframe Extraction (`nlargest`) |
| **Maximum Salaries** | Peak compensation limits within each department | Groupby Aggregation (`max`) |
| **Above-Average Base** | Count of employees exceeding organizational mean | Boolean Filtering |
| **Experience Trends** | Average seniority levels grouped by department | Groupby Aggregation (`mean`) |
| **Age Spread** | Distribution of workforce age demographics | Histogram with KDE |

## Result Interpretation & Learnings
*   **Holistic Data Understanding:** The sequential analytical operations yield a comprehensive view of the employee dataset structures.
*   **Visual Trend Identification:** Generated visualizations effectively highlight salary disparities across departments and demographic compositions.
*   **Variable Correlation:** Scatter plots confirm and illustrate the direct correlation between employee experience and compensation levels.
*   **Proficiency in Aggregation:** The implementation reinforces practical skills utilizing core Pandas aggregation functions, including `groupby`, `value_counts`, and `nlargest`.
*   **Data Translation:** The process demonstrates the application of Seaborn and Matplotlib in converting raw tabular data into interpretable visual narratives.
