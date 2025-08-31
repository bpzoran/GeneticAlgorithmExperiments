import matplotlib.pyplot as plt


def fitness_per_generation_plot(min_cost_per_generation, title, color):
    plt.plot(min_cost_per_generation, color=color)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.show()
