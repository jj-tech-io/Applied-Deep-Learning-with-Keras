import torch.nn as nn
from torch.autograd import Variable
import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchviz import make_dot
# from torchview import draw_graph

import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL
from PIL import Image
mpl.style.use('seaborn-deep')
# CMAP_SPECULAR = cm.get_cmap("cividis")
CMAP_SPECULAR = cm.get_cmap("viridis")
#use gpu if available




class Autoencoder(nn.Module):
    device = None
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(3,80),
            nn.ReLU(),
            nn.Linear(80,80),
            nn.ReLU(),
            nn.Linear(80,80),
            nn.ReLU(),
            nn.Linear(80,5),

        )
        #add layer names 
        self.encoder[0].name = "encoder_in"
        self.encoder[1].name = "encoder_relu_1"
        self.encoder[2].name = "encoder_dense_1"
        self.encoder[3].name = "encoder_relu_2"
        self.encoder[4].name = "encoder_dense_2"
        self.encoder[5].name = "encoder_relu_3"
        self.encoder[6].name = "encoder_out"

        
        self.decoder = nn.Sequential(
            nn.Linear(5,80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 3)

        )
        #add layer names
        self.decoder[0].name = "decoder_in"
        self.decoder[1].name = "decoder_relu_1"
        self.decoder[2].name = "decoder_dense_1"
        self.decoder[3].name = "decoder_relu_2"
        self.decoder[4].name = "decoder_dense_2"
        self.decoder[5].name = "decoder_relu_3"
        self.decoder[6].name = "decoder_out"
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)


    def backward(self, loss):
        loss.backward()

    def forward(self, pix, param):
        x = self.encoder(pix)
        y = self.decoder(param)
        ae = self.decoder(self.encoder(pix).reshape(-1,5).float())
        return x, y, ae
    def optimize(self, optimizer):
        self.optimizer.step()
        optimizer.zero_grad()
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    # def show_graph(self, model, input, output, device):
    #     batch_size = 2
    #     # device='meta' -> no memory is consumed for visualization
    #     # model_graph = draw_graph(model, input_size=(1,3), device=self.device)
    #     # model_graph.visual_graph
    #     make_dot(model(input, output), params=dict(model.named_parameters())).render("autoencoder", format="png")
        
def load_data():
    np.random.seed(42)
    #load csv into 
    # data_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\EncoderDecoder\JJ_LUT.csv"
    headers = "Cm,Ch,Bm,Bh,T,sR,sG,sB"
    # df = pd.read_csv(config.LUTv1_PATH, sep=",", header=None, names=headers.split(","))
    # df = pd.read_csv(config.LUTv2_PATH, sep=",", header=None, names=headers.split(","))
    # df = pd.read_csv(config.LUTv3_PATH, sep=",", header=None, names=headers.split(","))

    df = pd.read_csv(r"C:\Users\joeli\OneDrive\Documents\GitHub\Applied-Deep-Learning-with-Keras\data\JJ_LUTv2.csv", sep=",", header=None, names=headers.split(","))

    df.head()
    #remove header
    df = df.iloc[1:]
    #inputs = Cm,Ch,Bm,epi_thick
    y = df[['Cm','Ch','Bm','Bh','T']]
    print(y.head())

    #outputs = sR,sG,sB
    x = df[['sR','sG','sB']]
    print(x.head())

    df.head()
    #remove headers and convert to numpy array
    x = df[['sR','sG','sB']].iloc[1:].to_numpy()
    y = df[['Cm','Ch','Bm','Bh','T']].iloc[1:].to_numpy()
    
    #train nn on x,y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    #create data loader for training
    
    #numpy arrays
    x_train = np.asarray(x_train).reshape(-1,3).astype('float32')


    x_test = np.asarray(x_test).reshape(-1,3).astype('float32')

    print(f"bef norm x_train[0] {x_train[0]}")
    #normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(f"aft norm x_train[0] {x_train[0]}")
    return x_train, x_test, y_train.astype('float32'), y_test.astype('float32')

def train():
    net = Autoencoder()
    encoder = net.encoder
    decoder = net.decoder
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # , loader, loss_func, optimizer
    x_train, x_test, y_train, y_test = load_data()
    print(f"x_train {x_train.shape}")
    #convert to tensor
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    train = list(zip(x_train, y_train))
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=64, shuffle=True)
    print(f"train_loader {train_loader}")

    test = list(zip(x_test, y_test))
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=64, shuffle=True)
    batch_size = 64
    net.train()
    loss_vals = []
    epochs = 5
    for e in range(0, epochs):
        for pixels,parameters in train:
            #forward pass
            pixels = pixels.float()
            parameters = parameters.reshape(-1,5).float()
            x, y, end_to_end_pred = net.forward(pixels, parameters)
            #calculate loss
            loss = loss_func(end_to_end_pred, pixels)
            loss2 = loss_func(pixels, y)
            loss3 = loss_func(parameters, x )
            loss_total = (loss + loss2 + loss3)/3.0
            if e % 100 == 0:
                print(f"actual {pixels} end_to_end_pred {end_to_end_pred}")
                print(f"loss {loss_total.item()}")
            #backprop
            optimizer.zero_grad()
            net.backward(loss_total)
            optimizer.step()
            loss_vals.append(loss_total.item())
            print(f"loss {loss_total.item()}")
            
    return loss_vals, net


#entry point
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    delay = 1.5
    #delay 
    time.sleep(delay)

    losses, net = train()
    net.device = device
    
    

    plt.plot(losses)
    plt.show()
    x_train, x_test, y_train, y_test = load_data()
    #predict x_test
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    test = list(zip(x_test, y_test))
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=64, shuffle=True)
    net.eval()
    for pixels,parameters in test:
        #predict
        pixels = pixels.reshape(-1,3).float()
        parameters = parameters.reshape(-1,5).float()
        x, y, end_to_end_pred = net.forward(pixels, parameters)
        # print(f"actual: {pixels} end_to_end_pred {end_to_end_pred}" )
        #load image
    net = net.to(device)
    decoder = net.decoder.to(device)
    encoder = net.encoder.to(device)
    image_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\EncoderDecoder\Results\Assets\Metina\XYZ_albedo_lin_srgb.png"

    img = Image.open(image_path)
    #convert to RGB
    img = img.convert('RGB')
    im = img.resize((2048, 2048))
    img = np.asarray(img)
    image_original = img.copy()
    print(f"img {img.shape}")
    #convert to tensor
    img = torch.from_numpy(img/255.0)
    img = img.float()
    #reshape

    print(f"img {img.shape}")
    #predict
    parameters = net.encode(img)
    print(f"parameters {parameters.shape}")
    #image plot parameters
    recovered_img = net.decode(parameters).detach().numpy()
    print(f"recovered_img {recovered_img.shape}")
    recovered_img = recovered_img.reshape(2048,2048,3)*255.0
    plt.imshow(recovered_img)
    plt.show()

    parameters = parameters.detach().numpy()
    c_m = parameters[:,:,0]
    ch = parameters[:,:,1]
    bm = parameters[:,:,2]
    bh = parameters[:,:,3]
    t = parameters[:,:,4]

    print(f"cm {c_m.shape}")
    print(f"ch {ch.shape}")
    print(f"bm {bm.shape}")
    print(f"bh {bh.shape}")
    print(f"t {t.shape}")


    #plot all parameters as images side by side
    fig, axs = plt.subplots(1,5)
    axs[0].imshow(c_m, cmap=CMAP_SPECULAR)
    axs[0].set_title("cm")
    plt.colorbar(axs[1].imshow(c_m, cmap=CMAP_SPECULAR), ax=axs[1],fraction=0.046, pad=0.04)
    axs[1].set_title("cm")
    axs[2].imshow(ch, cmap=CMAP_SPECULAR)
    axs[2].set_title("ch")
    plt.colorbar(axs[2].imshow(ch, cmap=CMAP_SPECULAR), ax=axs[2],fraction=0.046, pad=0.04)
    axs[3].imshow(bm, cmap=CMAP_SPECULAR)
    axs[3].set_title("bm")
    plt.colorbar(axs[3].imshow(bm, cmap=CMAP_SPECULAR), ax=axs[3],fraction=0.046, pad=0.04)
    axs[4].imshow(bh, cmap=CMAP_SPECULAR)
    axs[4].set_title("bh")
    plt.colorbar(axs[4].imshow(bh, cmap=CMAP_SPECULAR), ax=axs[4],fraction=0.046, pad=0.04)
    plt.show()


    print(f"recovered_img {recovered_img.shape}")
    print(f"max {np.max(recovered_img)}")
    print(f"min {np.min(recovered_img)}")
    print(f"mean {np.mean(recovered_img)}")
    print(f"std {np.std(recovered_img)}")
    
