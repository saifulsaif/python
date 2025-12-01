import matplotlib.pyplot as plt
import numpy as np

import numpy as np

# Example: Market share or product sales distribution
y = np.array([420, 310, 280, 190, 160, 140, 110, 90, 75, 55])
mylabels = [
    "Product A",
    "Product B",
    "Product C",
    "Product D",
    "Product E",
    "Product F",
    "Product G",
    "Product H",
    "Product I",
    "Product J"
]


plt.pie(y, labels = mylabels)
plt.legend()
plt.show()


plt.show(block=False)  # Non-blocking display

# You can continue in terminal while plot is open
print("Plot displayed! You can continue working...")

# Keep the plot open (optional)
plt.pause(0.001)  # Small pause to ensure plot renders
input("Press Enter to close the plot...")  # Wait for user input
plt.close()