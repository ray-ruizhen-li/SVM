import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score
#to build SVM model
from sklearn.svm import SVC

#Metrics to evaluate the model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#for tuning the model
from sklearn.model_selection import GridSearchCV


# Plot cluster method scatter chart
def plot_scatter(df):
    sns.scatterplot(x='Annual_Income', y = 'Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.show()

# Plot elbow chart
def elbow(df, start_range, end_range, *args):
    # try using a for loop
    k = range(start_range,end_range)
    K = []
    WCSS = []
    column = []
    for items in args:
        column.append(items)
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df[column])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)
    # Store the number of clusters and their respective WSS scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
    # Now, plot a Elbow plot
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()

def silhouette(df, start_range, end_range, *args):
    # same as above, calculate sihouetter score for each cluster using a for loop
    k = range(start_range,end_range)
    K = []
    ss = []
    column = []
    for items in args:
        column.append(items)
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[column])
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[column], ypred)
        K.append(i)
        ss.append(sil_score)
    # Store the number of clusters and their respective silhouette scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':ss})
    # Now, plot a Elbow plot
    wss.plot(x='cluster', y = 'WSS_Score')
    plt.xlabel('No. of clusters')
    plt.ylabel('WSS Score')
    plt.title('Elbow Plot')
    plt.show()

#creating metric function
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()