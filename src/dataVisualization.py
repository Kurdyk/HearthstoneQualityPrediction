import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataEncoding import *

def plot_data(data,n_dim):
    if n == 1:
        plt.plot(data)
        plt.show()
    elif n == 2:
        plt.scatter(data[:,0], data[:,1])
        plt.show()
    elif n == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2])
        plt.show()
    else:
        print("Dimension not supported for plotting")

if __name__ == "__main__":
	df = pd.read_csv("../hearthstone.csv")
	de = DataEncoder(df)
	resulting_df = de.encode()

	n_dim = 2
	plot_data(resulting_df,n_dim)