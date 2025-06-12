import matplotlib.pyplot as plt
from datetime import datetime

knots2ms = 0.5144


def plot_vel(filename):
    uv_values = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        skip_next = False
        
        for line in lines:
            columns = line.strip().split()
            
            if columns[0] == 'WD':
                if skip_next:
                    skip_next = False
                else:
                    skip_next = True
                continue
            
            if columns[0] == 'UV':
                uv_values.append(float(columns[3]))
    
    uv_values_converted = [value * knots2ms for value in uv_values]
    plt.plot(uv_values_converted, marker='o')
    plt.xlabel('Index')
    plt.ylabel('UV (m/s)')
    plt.title('UV from mesoanalysis at point 48.986708, -64.455250')
    plt.show()

# Replace 'your_file.txt' with the path to your .txt file
plot_vel('UV_48.986708_-64.455250.txt')
