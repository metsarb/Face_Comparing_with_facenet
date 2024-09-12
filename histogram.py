import matplotlib.pyplot as plt


pos_distances = [0.94,
                 0.65,
                 0.68,
                 0.85,
                 0.95,
                 0.79,
                 0.75,
                 0.70,
                 0.69,
                 0.69,
                 0.53,
                 0.78,
                 0.60,
                 0.87,
                 0.43,
                 0.95,
                 0.97,
                 0.75,
                 0.61,
                 0.59,
                 0.75,
                 0.68,
                 0.72,
                 0.64,
                 0.56,
                 0.89,
                 0.94,
                 0.57,
                 0.77,
                 0.80,
                 0.45,
                 0.58,
                 0.85,
                 0.81,
                 0.89,
                 0.78,
                 0.71,
                 0.67,
                 0.70,
                 0.57,
                 0.60,
                 0.68,
                 0.74,
                 0.43,
                 0.76,
                 0.83,
                 0.53,
                 0.64,
                 0.59
                 ]

neg_distances = [1.49,
                 1.57,
                 1.22,
                 1.22,
                 1.32,
                 1.32,
                 1.37,
                 1.33,
                 1.34,
                 1.39,
                 1.47,
                 1.47,
                 1.00,
                 1.31,
                 1.50,
                 1.51,
                 1.37,
                 1.52,
                 1.42,
                 1.33,
                 1.43,
                 1.41,
                 1.28,
                 1.40,
                 1.45,
                 1.25,
                 1.29,
                 1.51,
                 1.38,
                 1.47,
                 1.33,
                 1.27,
                 1.41,
                 1.11,
                 1.42,
                 1.44,
                 1.46,
                 1.16,
                 1.03,
                 1.06,
                 1.43,
                 1.56,
                 1.34,
                 1.06,
                 1.42,
                 1.27,
                 1.06,
                 1.42,
                 1.58,
                 1.39,
                 1.27,
                 1.34,
                 1.27,
                 1.42,
                 1.44,
                 1.38,
                 1.52,
                 1.51,
                 1.38,
                 1.42,
                 1.49,
                 1.19,
                 1.20,
                 1.30,
                 1.22,
                 1.47,
                 1.60,
                 1.46,
                 1.39,
                 1.43,
                 1.43,
                 1.45,
                 1.53,
                 1.37,
                 1.40,
                 1.35,
                 1.13,
                 1.31,
                 1.44,
                 1.25,
                 1.38,
                 1.26,
                 1.32,
                 1.00,
                 1.41,
                 1.28,
                 1.54,
                 1.37,
                 1.27,
                 1.42,
                 1.18,
                 1.34,
                 1.27,
                 1.15,
                 1.52,
                 1.25,
                 1.12,
                 1.72,
                 1.87,
                 1.76
                 ]

plt.figure(figsize = (10,6))

plt.hist(pos_distances, bins = 20, alpha = 0.5, label = 'Positive', color = 'blue')

plt.hist(neg_distances, bins = 20, alpha = 0.5, label = 'Negative', color = 'red')

plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('The distance of positive and negative face pairs')
plt.legend()

plt.show()


