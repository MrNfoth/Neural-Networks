import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List

# --- Константы для выбора функции активации ---
SIGMOID = 'sigmoid'
RELU = 'relu'

# --- Реализации функций активации и их производных ---
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Сигмоида"""
    # ДОБАВЛЕНО 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Производная сигмоида"""
    # ДОБАВЛЕНО
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU (Rectified Linear Unit)"""
    # ДОБАВЛЕНО
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Производная ReLU"""
    # ДОБАВЛЕНО >>>
    return (x > 0).astype(float)

# --- Сопоставление констант с функциями активации ---
activation_map: dict = {
    SIGMOID: (sigmoid, sigmoid_derivative),
    RELU: (relu, relu_derivative)
}

class MLP:
    def __init__(
        self,
        layer_sizes: List[int],
        activation_name: str,
        learning_rate: float = 0.1
    ) -> None:
        assert activation_name in activation_map, f"Неподдерживаемая активация: {activation_name}"
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        
        self.activation_fn, self.activation_derivative_fn = activation_map[activation_name]
        
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for in_neurons, out_neurons in zip(layer_sizes[:-1], layer_sizes[1:]):
            weight_matrix = np.random.randn(out_neurons, in_neurons) * 0.1
            bias_vector = np.zeros((out_neurons, 1))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.activations: List[np.ndarray] = [inputs]
        self.z: List[np.ndarray] = []
        
        last_layer_idx = len(self.layer_sizes) - 1
        
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = W @ self.activations[idx] + b
            self.z.append(Z)
            
            # последняя активация не через ReLU/Sigmoid, а линейная
            A = self.activation_fn(Z) if idx != last_layer_idx - 1 else Z
            self.activations.append(A)
        return sigmoid(self.activations[-1])

    def backward(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Обратное распространение ошибки и обновление весов
        """
        # ДОБАВЛЕНО
        m = targets.shape[1]  # количество примеров

        # Вычисляем выход сети (предсказание)
        predictions = self.forward(inputs)

        # Ошибка на выходе (MSE -> dL/dA)
        dA = -(targets - predictions)

        # Последний слой (сигмоида)
        dZ = dA * sigmoid_derivative(self.activations[-1])
        dW = (1 / m) * (dZ @ self.activations[-2].T)
        dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Обновляем последний слой
        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * dB

        # Проходим по скрытым слоям в обратном порядке
        for l in reversed(range(len(self.weights) - 1)):
            dA = self.weights[l + 1].T @ dZ
            dZ = dA * self.activation_derivative_fn(self.z[l])
            dW = (1 / m) * (dZ @ self.activations[l].T)
            dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            
            self.weights[l] -= self.learning_rate * dW
            self.biases[l] -= self.learning_rate * dB

    def train(self, inputs: np.ndarray, targets: np.ndarray,
              epochs: int = 1000, print_interval: int = 100) -> None:
        for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
            predictions = self.forward(inputs)
            loss = np.mean((targets - predictions) ** 2)
            
            self.backward(inputs, targets)
            
            if epoch == 1 or epoch % print_interval == 0:
                predicted_labels = (predictions > 0.5).astype(int)
                accuracy = np.mean(predicted_labels == targets)
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}%")

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

if __name__ == "__main__":
    sample_count = 2000
    bottom_left = np.random.uniform(0, 0.5, (sample_count, 2))
    top_right = np.random.uniform(0.5, 1.0, (sample_count, 2))
    top_left = np.random.uniform(0, 0.5, (sample_count, 2))
    top_left[:,1] += 0.5
    bottom_right = np.random.uniform(0.5, 1.0, (sample_count, 2))
    bottom_right[:,1] -= 0.5

    class_zero = np.vstack((bottom_left, top_right))
    class_one = np.vstack((top_left, bottom_right))
    
    data = np.vstack((class_zero, class_one))
    labels = np.concatenate((np.zeros((2*sample_count, 1)), np.ones((2*sample_count, 1))), axis=0)
    
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_point = int(num_samples * 0.75)
    train_idx = indices[:split_point]
    test_idx  = indices[split_point:]
    train_data, train_labels = data[train_idx], labels[train_idx]
    test_data,  test_labels  = data[test_idx],  labels[test_idx]

    X_train, Y_train = train_data.T, train_labels.T
    X_test,  Y_test  = test_data.T,  test_labels.T

    output_folder = 'outputs'
    os.makedirs(output_folder, exist_ok=True)

    plt.scatter(class_zero[:,0], class_zero[:,1], c='red', label='Класс 0')
    plt.scatter(class_one[:,0], class_one[:,1], c='blue', label='Класс 1')
    plt.legend(loc='upper right')
    plt.title("Исходные данные")
    plt.savefig(f"{output_folder}/source_data.png")
    plt.show()
    
    model = MLP(
        layer_sizes=[2, 10, 10, 1],
        activation_name=RELU,
        learning_rate=0.1
    )
    model.train(X_train, Y_train, epochs=10000, print_interval=1000)
    
    predictions = model.predict(X_test).reshape(-1)
    pred_colors = ['blue' if p > 0.5 else 'red' for p in predictions]
    plt.scatter(test_data[:,0], test_data[:,1], c=pred_colors)
    plt.title("Предсказанные данные")
    plt.savefig(f"{output_folder}/predicted_data.png")
    plt.show()
    
    error_colors = ['black' if ((predictions[j] > 0.5) != labels[i,0]) else
                    ('blue' if predictions[j] > 0.5 else 'red')
                    for i, j in zip(test_idx, range(len(test_idx)))]
    plt.scatter(test_data[:,0], test_data[:,1], c=error_colors)
    plt.title("Ошибки при классификации (чёрный)")
    plt.savefig(f"{output_folder}/errors.png")
    plt.show()
