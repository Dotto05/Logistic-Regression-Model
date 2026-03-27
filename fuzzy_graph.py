import numpy as np
import matplotlib.pyplot as plt

# ==============================
# AGE
# ==============================
age = np.linspace(0, 80, 100)

young = np.maximum(0, np.minimum((40 - age)/40, 1))
middle = np.maximum(0, np.minimum((age - 30)/20, (60 - age)/20))
old = np.maximum(0, np.minimum((age - 50)/30, 1))

plt.figure()
plt.plot(age, young, label="Young")
plt.plot(age, middle, label="Middle")
plt.plot(age, old, label="Old")
plt.title("Fuzzy Membership - Age")
plt.xlabel("Age")
plt.ylabel("Membership")
plt.legend()
plt.grid()

# ==============================
# CHOLESTEROL
# ==============================
chol = np.linspace(100, 400, 100)

low = np.maximum(0, np.minimum((200 - chol)/100, 1))
medium = np.maximum(0, np.minimum((chol - 150)/50, (250 - chol)/50))
high = np.maximum(0, np.minimum((chol - 200)/100, 1))

plt.figure()
plt.plot(chol, low, label="Low")
plt.plot(chol, medium, label="Medium")
plt.plot(chol, high, label="High")
plt.title("Fuzzy Membership - Cholesterol")
plt.xlabel("Cholesterol")
plt.ylabel("Membership")
plt.legend()
plt.grid()

# ==============================
# BLOOD PRESSURE
# ==============================
bp = np.linspace(80, 180, 100)

normal = np.maximum(0, np.minimum((120 - bp)/40, 1))
elevated = np.maximum(0, np.minimum((bp - 110)/20, (140 - bp)/20))
high_bp = np.maximum(0, np.minimum((bp - 130)/50, 1))

plt.figure()
plt.plot(bp, normal, label="Normal")
plt.plot(bp, elevated, label="Elevated")
plt.plot(bp, high_bp, label="High")
plt.title("Fuzzy Membership - Blood Pressure")
plt.xlabel("BP")
plt.ylabel("Membership")
plt.legend()
plt.grid()

plt.show()