import torch
#torch is treated as an object with pytorch imported for some reason

# 1. Training examples: each row contains one input or answer.
# y=3x+2

#
x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [5.0], [8.0], [11.0], [14.0]])

# 2. Model: prediction = weight * input + bias.
# The weight and bias start with random values.
model = torch.nn.Linear(1, 1)

# 3. Measure prediction error and choose how to update the model.
# MSE = mean squared error, loss is how far off from expected
loss_function = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. Learn by repeatedly predicting, measuring, and adjusting.
for epoch in range(1000):
    #Use model's current weight & bias to calculate answers for inputs x
    prediction = model(x)
    #compare to correct
    loss = loss_function(prediction, y)

    optimizer.zero_grad()  # Clear gradients from the previous step.
    loss.backward()       # Calculate gradients for weight and bias.
    #Gradients are calculated from taking the derivative of the mean standard error loss formula
    
    #essentially calc's optimization stuff
    optimizer.step()      # Adjust weight and bias to reduce error.

    #step basically juts applies new weight & bias *= learning rate * gradient
    if epoch % 200 == 0:
        print(f"Step {epoch}: loss = {loss.item():.6f}")

# 5. Inspect what it learned and try an input absent from training.
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

with torch.no_grad():
    result = model(torch.tensor([[5.0]]))
    print("Prediction for x = 5:", result.item())