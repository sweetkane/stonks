from common.common_imports import *




def get_grad_mean(model: torch.nn.Module, device):
    means = torch.empty((0,), dtype=float).to(device)
    for param in model.parameters():
        mean = torch.mean(torch.abs(param.grad)).unsqueeze(0)
        means = torch.cat((means, mean))

    return torch.mean(means)

def get_grad_max(model: torch.nn.Module, device):
    maxes = torch.empty((0,), dtype=float).to(device)
    for param in model.parameters():
        max = torch.max(torch.abs(param.grad)).unsqueeze(0)
        maxes = torch.cat((maxes, max))

    return torch.max(maxes)

def plot_loss(y):
    x = np.arange(len(y))
    plt.scatter(x, y)
    z = np.polyfit(x, np.log10(y), 1)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, 10**p(x), "r--")
    plt.yscale("log")
    plt.show
    print(p)

def print_tensors(msg):
    print(f"\nvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n{msg}\n")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.is_cuda, obj.names)
        except:
            pass
    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

def count_tensors(msg):
    num_tensors = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                num_tensors += 1
        except:
            pass
    print(f"num_tensors {msg}:", num_tensors)
