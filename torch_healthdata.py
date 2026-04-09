import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from pandas import to_datetime


def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


class StackedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=4, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# fix random seed for reproducibility
np.random.seed(67)
torch.manual_seed(67)

# load the dataset
# we pick participant one for example
dataframe = read_csv("pmdata/p01/googledocs/reporting.csv", engine="python")

# Convert date to datetime with mm/dd/yyyy format
dataframe['date'] = to_datetime(dataframe['date'], format='%d/%m/%Y')

# Sort by date
dataframe = dataframe.sort_values('date').reset_index(drop=True)

# Normalize date: set lowest date to 0 (as number of days from minimum)
dataframe['date'] = (dataframe['date'] - dataframe['date'].min()).dt.days

# Convert weight to float32
dataframe['weight'] = dataframe['weight'].astype('float32')

# Skip rows with missing weight values
dataframe = dataframe.dropna(subset=['weight']).reset_index(drop=True)

dataset = dataframe['weight'].values.astype("float32").reshape(-1, 1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
train, test = dataset[0:train_size, :], dataset[train_size : len(dataset), :]

# reshape into X=t and Y=t+1
look_back = 3
train_x, train_y = create_dataset(train, look_back)
test_x, test_y = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

# convert to tensors
train_x_t = torch.tensor(train_x, dtype=torch.float32)
train_y_t = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
test_x_t = torch.tensor(test_x, dtype=torch.float32)

# create and fit the LSTM network
model = StackedLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# early stopping config
max_epochs = 500
patience = 25
rmse_min_delta = 1e-4
loss_gain_min_delta = 1e-6

best_rmse = float("inf")
best_loss = float("inf")
stagnation_epochs = 0

model.train()
for epoch in range(max_epochs):
    optimizer.zero_grad()
    output = model(train_x_t)
    loss = criterion(output, train_y_t)
    loss.backward()
    optimizer.step()

    # RMSE in normalized space for early-stopping feedback.
    current_loss = loss.item()
    current_rmse = float(np.sqrt(current_loss))

    rmse_gain = best_rmse - current_rmse
    loss_gain = best_loss - current_loss
    rmse_improved = rmse_gain > rmse_min_delta
    loss_improved = loss_gain > loss_gain_min_delta

    if rmse_improved:
        best_rmse = current_rmse
    if loss_improved:
        best_loss = current_loss

    if not rmse_improved and not loss_improved:
        stagnation_epochs += 1
    else:
        stagnation_epochs = 0

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{max_epochs}, "
            f"Loss: {current_loss:.6f}, RMSE: {current_rmse:.6f}, "
            f"stagnation: {stagnation_epochs}/{patience}"
        )

    if stagnation_epochs >= patience:
        print(
            f"Early stopping at epoch {epoch + 1}: "
            f"RMSE and optimizer gain stagnated for {patience} epochs."
        )
        break

# make predictions
model.eval()
with torch.no_grad():
    train_predict = model(train_x_t).numpy()
    test_predict = model(test_x_t).numpy()

# invert predictions
train_predict = scaler.inverse_transform(train_predict)
train_y_inv = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y_inv = scaler.inverse_transform([test_y])

# calculate root mean squared error
train_score = np.sqrt(mean_squared_error(train_y_inv[0], train_predict[:, 0]))
print("Train Score: %.2f RMSE" % train_score)
test_score = np.sqrt(mean_squared_error(test_y_inv[0], test_predict[:, 0]))
print("Test Score: %.2f RMSE" % test_score)

# shift train predictions for plotting
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back : len(train_predict) + look_back, :] = train_predict

# shift test predictions for plotting
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[
    len(train_predict) + (look_back * 2) + 1 : len(dataset) - 1, :
] = test_predict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label='Actual Data')
plt.plot(train_predict_plot, label='Train Predictions')
plt.plot(test_predict_plot, label='Test Predictions')
plt.title('Weight Predictions vs Actual Data')
plt.xlabel('Days from Start')
plt.ylabel('Weight')
plt.legend()
plt.savefig("predictions.png")
plt.close()

# save the model
torch.save(model.state_dict(), "model.pth")

# load the model
# model = StackedLSTM()
# model.load_state_dict(torch.load("model.pth"))

# # make predictions
# model.eval()
# with torch.no_grad():
#     test_predict = model(test_x_t).numpy()