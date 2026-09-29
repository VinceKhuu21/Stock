import torch

# 1. Training examples: each row contains one input or answer.
x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [5.0], [8.0], [11.0], [14.0]])

# 2. Model: prediction = weight * input + bias.
# The weight and bias start with random values.
model = torch.nn.Linear(1, 1)

# 3. Measure prediction error and choose how to update the model.
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Learn by repeatedly predicting, measuring, and adjusting.
for epoch in range(1000):
    prediction = model(x)
    loss = loss_function(prediction, y)

    optimizer.zero_grad()  # Clear gradients from the previous step.
    loss.backward()       # Calculate gradients for weight and bias.
    optimizer.step()      # Adjust weight and bias to reduce error.

    if epoch % 200 == 0:
        print(f"Step {epoch}: loss = {loss.item():.6f}")

# 5. Inspect what it learned and try an input absent from training.
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

with torch.no_grad():
    result = model(torch.tensor([[5.0]]))
    print("Prediction for x = 5:", result.item())