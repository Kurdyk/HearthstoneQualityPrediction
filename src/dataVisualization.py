import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataEncoding import *

def viz_3D(data):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	x = data[:,0]
	y = data[:,1]
	z = data[:,2]

	ax.scatter(x, y, z)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('3D visualization')

	plt.show()

if __name__ == "__main__":
	df = pd.read_csv("../OldData/hearthstone.csv")
	de = DataEncoder(df)

	test_corpus = ["Discover a minion. Give it +1/+1.",
		"Enrage: +2 Attack.",
		"Deathrattle: Summon a random friendly Beast that died this game.",
		"At the end of your turn, eat a random enemy minion and gain its stats.",
		"Taunt. Deathrattle: Deal 2 damage to ALL characters.",
		"Discover a 6-Cost minion. Summon it with Taunt and Divine Shield.",
		"Spell Damage +2 Deathrattle:The next minion you draw inherits these powers.",
		"Deal damage to a minion equal to its Attack.",
		"Draw 2 cards. Costs (2) less for each Treant you control.",
		"Destroy ALL odd-Attack minions.",
		"Deal 3 damage. Your next Hero Power deals 2 more damage.",
		"Can't be targeted by spells or Hero Powers.",
		"Charge, Divine Shield, Taunt, Windfury"
		]

	resulting_df = de.encode(test_corpus,3)

	viz_3D(resulting_df)