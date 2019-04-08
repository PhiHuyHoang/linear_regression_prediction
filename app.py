import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv_file(csv_file_name):
    csv_dict = {}
    with open(csv_file_name, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for key,row in enumerate(reader):
            csv_dict[key] = row
    csvFile.close()
    return csv_dict

def round_up(number_to_round):
    n = 0
    rem = number_to_round % 10
    if rem < 5:
        n = int(number_to_round / 10) * 10
    else:
        n = int((number_to_round + 10) / 10) * 10
    return n

total_dict = read_csv_file("500_Person_Gender_Height_Weight_Index.csv")

male_list = []
female_list = []
for k,v in total_dict.items():
    if v[0] == "Male":
        male_list.append(v)
    else:
        female_list.append(v)

men_height = []
men_weight = []
for item in male_list:
    men_height.append(int(item[1]))
    men_weight.append(int(item[2]))

men_height_min = round_up(min(men_height))
men_height_max = round_up(max(men_height))
men_weight_min = round_up(min(men_weight))
men_weight_max = round_up(max(men_weight))

men_height.sort()
men_weight.sort()

# height (cm)
x_men_height = np.array([men_height]).T
# weight (kg)
y_men_weight = np.array([men_weight]).T
# Visualize data
plt.plot(x_men_height, y_men_weight, 'ro')
plt.axis([men_height_min, men_height_max, men_weight_min, men_weight_max])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

# Building Xbar
one = np.ones((x_men_height.shape[0], 1))
Xbar = np.concatenate((one, x_men_height), axis = 1)

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y_men_weight)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 200, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line
plt.plot(x_men_height, y_men_weight, 'ro')
plt.plot(x0,y0)
plt.axis([men_height_min, men_height_max, men_weight_min, men_weight_max])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

y1 = w_1*men_height[0] + w_0
y2 = w_1*men_height[100] + w_0

y1_real = men_weight[0]
y2_real = men_weight[100]

print( u'Predict weight of person with height %d cm: %.2f (kg), real number: %d (kg)'  %(men_height[0],y1,y1_real))
print( u'Predict weight of person with height %d cm: %.2f (kg), real number: %d (kg)'  %(men_height[100],y2,y2_real))
