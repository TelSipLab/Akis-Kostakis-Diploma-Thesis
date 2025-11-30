"""
Hello World LSTM
=================
The simplest possible LSTM example: Learn to count!

Task: Given input [0, 1, 2], predict [1, 2, 3]
Same task as RNN but with LSTM architecture
"""

import torch
import torch.nn as nn

# Create simple training data: learn to add 1 to each number
X = torch.tensor([[[0.], [1.], [2.], [3.], [4.]]])  # Shape: (batch=1, seq_len=5, features=1)
y = torch.tensor([[[1.], [2.], [3.], [4.], [5.]]])  # Target: each number + 1

# Define a simple LSTM
class HelloLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=10, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # Process sequence with LSTM (has cell state + hidden state)
        out = self.fc(out)      # Map to output
        return out

# Create model, loss, optimizer
model = HelloLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
print("Training LSTM to learn: output = input + 1")
print("-" * 40)
for epoch in range(100):
    # Forward pass
    prediction = model(X)
    loss = criterion(prediction, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Test
print("\n" + "=" * 40)
print("Testing LSTM:")
print("=" * 40)
test_input = torch.tensor([[[0.], [1.], [2.], [3.], [4.]]])
model.eval()
with torch.no_grad():
    output = model(test_input)

print("Input:  ", test_input.squeeze().tolist())
print("Output: ", [round(x, 2) for x in output.squeeze().tolist()])
print("Target: ", y.squeeze().tolist())
print("\nLSTM learned to add 1 to each number!")

print("\n" + "=" * 40)
print("RNN vs LSTM:")
print("=" * 40)
print("RNN:  Only has hidden state (short-term memory)")
print("LSTM: Has cell state + hidden state + 3 gates")
print("      - Forget gate: what to forget")
print("      - Input gate:  what to remember")
print("      - Output gate: what to output")
print("\nLSTM is better for long sequences!")
