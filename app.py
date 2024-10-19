from flask import Flask, render_template, request, redirect, url_for
from markupsafe import Markup
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.cluster import KMeans
from kneed import KneeLocator
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and process apriori algorithm
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read the CSV file
        df = pd.read_csv(filepath, index_col=0)
        p = [0.01]*5 + [0.2]*14 + [0.05]*7 + [0.01]*8 + [0.05]*5

        # Normalize probabilities to sum to 1
        total = sum(p)
        p_normalized = [x / total for x in p]
        ages = np.random.choice(range(18, 57), size=df.shape[0], p=p_normalized)
        df['Age'] = ages

        # Step 3: Check for any missing values
        print(df.isnull().sum())

        

        def find_optimal_k_elbow(df, max_k=10):
            inertia = []
            k_values = range(1, max_k + 1)

            # Calculate inertia for each k
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=0)
                kmeans.fit(df[['Age']])  # Assuming you're clustering based on the 'Age' column
                inertia.append(kmeans.inertia_)

            # Find the "elbow" point using KneeLocator
            kn = KneeLocator(k_values, inertia, curve='convex', direction='decreasing')
            optimal_k = kn.elbow

            # Plot the Elbow graph
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, inertia, marker='o')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k')
            plt.grid(True)
            plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
            plt.legend()
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'Elbow_Method.png'))

            return optimal_k

        # Call the function before clustering and get the optimal k
        optimal_k = find_optimal_k_elbow(df)
        print(f'The optimal number of clusters is: {optimal_k}')

        # Step 4: Perform clustering based on age
        age_data = df[['Age']].values
        kmeans = KMeans(n_clusters=optimal_k)
        df['Age Cluster'] = kmeans.fit_predict(age_data)
        centroids = kmeans.cluster_centers_

        # Step 5: Association rule analysis for each cluster
        cluster_results = []
        all_rules = ""
        
        for cluster in df['Age Cluster'].unique():
            cluster_data = df[df['Age Cluster'] == cluster].drop(columns=['Age', 'Age Cluster'])
            frequent_itemsets = apriori(cluster_data, min_support=0.3, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
            
            if not rules.empty:
                # Find the rule with the largest support
                largest_rule = rules.loc[rules['support'].idxmax()]

                # Get the min and max age for the cluster
                age_min = df[df['Age Cluster'] == cluster]['Age'].min()
                age_max = df[df['Age Cluster'] == cluster]['Age'].max()

                # Print all rules for this cluster
                print(f"\nAssociation rules for Age Cluster {cluster} (Age range {age_min}-{age_max}):")
                for index, row in rules.iterrows():
                    antecedents = ', '.join(list(row['antecedents']))
                    consequents = ', '.join(list(row['consequents']))
                    rule_str = f"{antecedents} => {consequents}"
                    support = row['support']
                    print(f"Rule: {rule_str}, Support: {support:.4f}")
                    
                    all_rules += f"<li>Rule: {rule_str}, Support: {support:.4f}</li>"

                all_rules += "<br>"

                # Append the cluster result
                cluster_results.append({
                    'age_range': f"{age_min}-{age_max}",
                    'support': largest_rule['support'],
                    'rule': rule_str
                })

        # Sort cluster results by the minimum age (for ordered age ranges)
        cluster_results = sorted(cluster_results, key=lambda x: int(x['age_range'].split('-')[0]))
        all_rules = Markup(all_rules)
        # Visualization 1: Bar plot of largest support per age range
        plt.figure(figsize=(12, 8))  # Increase figure size for more space
        age_ranges = [res['age_range'] for res in cluster_results]
        supports = [res['support'] for res in cluster_results]
        rule_legends = [res['rule'] for res in cluster_results]

        bars = sns.barplot(x=age_ranges, y=supports, palette='coolwarm')
        plt.title('Support of Largest Association Rule per Age Range (Cluster)', pad=75, loc='center')
        plt.xlabel('Age Range')
        plt.ylabel('Support')

        # Add the association rules as the legend (remove frozenset)
        # for i, rule_legend in enumerate(rule_legends):
        #     # Position the text slightly higher than the bar height to avoid overlap
        #     plt.text(i, supports[i] + 0.02, rule_legend, ha='center', va='bottom', fontsize=10, rotation=45)

        # Extract the colors of the bars
        colors = [bar.get_facecolor() for bar in bars.patches]

        # Create a custom legend with the corresponding colors
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=rule_legends[i]) for i in range(len(rule_legends))]

        # Add the legend with custom colors
        plt.legend(handles=handles, title='Association Rules', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        sns.despine()
        # Adjust layout to make sure everything fits within the figure
        plt.tight_layout()
        

        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'Largest_Rules_Support.png'))

        # Visualization 2: Scatterplot of ages with cluster centroids
        # # Visualization 2: Scatterplot of ages with cluster centroids
        plt.figure(figsize=(10, 6))

        # Define a custom color palette for clusters (4 colors for 4 clusters)
        custom_palette = sns.color_palette('deep', n_colors=optimal_k)  # Replace with your desired colors

        # Scatterplot of Age data points colored by clusters
        scatter = sns.scatterplot(x=df['Age'], y=df['Age'], hue=df['Age Cluster'], palette=custom_palette, s=100)

        # Add centroids as 'X' markers
        centroids = df.groupby('Age Cluster')['Age'].mean().values
        for idx, centroid in enumerate(centroids):
            plt.scatter(centroid, centroid, marker='X', color='black', s=200, label='Centroid')

        plt.title('Clusters of Ages with Centroids', fontsize=14)
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Age', fontsize=12)

        # Set grid
        plt.grid(True)

        # Save the plot
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'Cluster_Age_Scatterplot.png'))



                
        # Render results
        return render_template('results.html',rules=all_rules,image0='Elbow_Method.png', image1='Largest_Rules_Support.png', image2='Cluster_Age_Scatterplot.png')


if __name__ == '__main__':
    app.run(debug=True)