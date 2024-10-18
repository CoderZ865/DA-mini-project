from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static\\uploads'

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

        ages = [random.randint(15, 60) for _ in range(df.shape[0])]
        df['Age'] = ages

        # Step 3: Check for any missing values
        print(df.isnull().sum())

        # Step 4: Perform clustering based on age
        age_data = df[['Age']].values
        kmeans = KMeans(n_clusters=4)
        df['Age Cluster'] = kmeans.fit_predict(age_data)

        # Step 5: Association rule analysis for each cluster
        cluster_results = {}
        rule_counts = []

        for cluster in df['Age Cluster'].unique():
            cluster_data = df[df['Age Cluster'] == cluster].drop(columns=['Age', 'Age Cluster'])
            frequent_itemsets = apriori(cluster_data, min_support=0.3, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
            cluster_results[cluster] = rules
            rule_counts.append(len(rules))
            
            print(f"Cluster {cluster} - Frequent Itemsets:")
            print(frequent_itemsets)

            if not rules.empty:
                print(f"\nCluster {cluster} - Association Rules:")
                pretty_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                print(pretty_rules.to_string(index=False))

        # Visualization of the results

        # Plot 1: Cluster Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Age Cluster', data=df, palette='viridis')
        plt.title('Number of Data Points in Each Age Cluster')
        plt.xlabel('Age Cluster')
        plt.ylabel('Count')
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'Cluster_Distribution_plot.png'))

        # Plot 2: Rule Counts per Cluster
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(range(len(rule_counts))), y=rule_counts, palette='magma')
        plt.title('Number of Association Rules per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Rules')
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'Rule_Counts.png'))
        
        # Render results
        return render_template('results.html', rules=rules.to_html(), image1='Cluster_Distribution_plot.png', image2='Rule_Counts.png')

if __name__ == '__main__':
    app.run(debug=True)