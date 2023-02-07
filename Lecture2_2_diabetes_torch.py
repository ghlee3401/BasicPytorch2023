import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

torch.manual_seed(777)
np.random.seed(777)

# Step 1 데이터 준비
diabetes = load_diabetes()
x = diabetes.data[:, 2]
y = diabetes.target

xt = torch.FloatTensor(x)  # numpy array를 torch의 tensor로 변환합니다. 
yt = torch.FloatTensor(y)

w = torch.ones([1], requires_grad=True)
b = torch.ones([1], requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=1e-2)
loss_fn = torch.nn.MSELoss()

# y_hat = xt[0] * w + b 
# loss = loss_fn(y_hat, yt[0])

# optimizer.zero_grad() 
# loss.backward() 
# optimizer.step() 

def plot(x, y, w, b, step):
    plt.scatter(x, y) 
    plt.xlabel("x")
    plt.ylabel("y")
    point1 = (-0.1, -0.1 * w.data.numpy() + b.data.numpy()) # x값이 적당히 작을때 직선 위의 점
    point2 = (0.15, 0.15 * w.data.numpy() + b.data.numpy()) # x값이 적당히 클때 직선 위의 점
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r')
    plt.grid() 
    plt.savefig("lec2_2_{}.png".format(step)) # 초기값일 때의 직선을 그립니다. 
    plt.close() 

cnt = 0 
idx = [i for i in range(len(x))]
for k in range(100):
    idx = np.random.permutation(idx)
    for i in idx:
        y_hat = xt[i] * w + b
        loss = loss_fn(y_hat, yt[i].unsqueeze(0))

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
    if k % 20 == 0:
        plot(x, y, w, b, cnt)
        cnt += 1

# plt.scatter(x, y) 
# plt.xlabel("x")
# plt.ylabel("y")
# point1 = (-0.1, -0.1 * w + b) # x값이 적당히 작을때 직선 위의 점
# point2 = (0.15, 0.15 * w + b) # x값이 적당히 클때 직선 위의 점
# plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r')
# plt.grid() 
# plt.savefig("lec2_1_3.png") # 초기값일 때의 직선을 그립니다. 
# plt.close() 

# x = 0.18
# y_pred = x * w + b

# print(y_pred)