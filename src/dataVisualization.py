import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from dataEncoding import *

def plot_data(df,n_dim):
	pca = PCA(n_components=n_dim)
	X_pca = pca.fit_transform(df)

    if n_dim == 1:
        plt.plot(X_pca)
        plt.show()
    elif n_dim == 2:
        plt.scatter(X_pca[:,0], X_pca[:,1])
        plt.show()
    elif n_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2])
        plt.show()
    else:
        print("Dimension not supported for plotting")

if __name__ == "__main__":
	df = pd.read_csv("../hearthstone.csv")
	de = DataEncoder(df)
	resulting_df = de.encode()

	n_dim = 2
	plot_data(resulting_df,n_dim)