import sys

from PetNet.CatsVsDogs import DogsVSCats


def main():
	dogVsCats = DogsVSCats("cuda:0")
	dogVsCats.make_training_data()
	dogVsCats.train()
	dogVsCats.test()
	sys.exit()


if __name__ == "__main__":
	main()
