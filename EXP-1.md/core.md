code:

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Input and output data for XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X, Y, epochs=1000, verbose=0)

# Evaluate the model
loss, acc = model.evaluate(X, Y, verbose=0)
print("Accuracy:", acc)

# Make predictions
predictions = model.predict(X)
print("Predictions:")
for i, p in enumerate(predictions):
    print(f"Input: {X[i]} => Predicted: {p[0]:.4f}")

# Plot the loss over epochs
plt.plot(history.history['loss'])
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

output:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/741e00e2-135c-4773-b5d8-cc9b96088d1d" />
