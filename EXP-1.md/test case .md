code:
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0,0,0])
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))            
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=1000, verbose=0)
loss, acc = model.evaluate(X, Y, verbose=0)
print("Accuracy:", acc)
predictions = model.predict(X)
print("Predictions:")
for i, p in enumerate(predictions):
    print(f"Input: {X[i]} => Predicted: {p[0]:.4f}")
plt.plot(history.history['loss'])
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
output
<img width="675" height="615" alt="image" src="https://github.com/user-attachments/assets/1fb5445c-5652-4705-8456-532c510591d3" />
