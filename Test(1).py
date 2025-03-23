import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
folder = "D:/Dataset/Galaxy_299491051364706304/images"
folder_array = os.listdir(folder)

image_data = []

for image in folder_array:
    hdul = fits.open(os.path.join(folder,image))
    image = hdul[0].data
    image_data.append(image)
    hdul.close()

H, W = image_data[0].shape
x_start,y_start = int(1023.5),int(743.499999999968)
cropped_data = []
for image in image_data:
    image = np.array(image[y_start:y_start+288,x_start:x_start+288])
    cropped_image = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1
    image = np.arcsin(cropped_image)/np.arcsin(np.max(cropped_image))
    cropped_data.append(image)
cropped_data = np.array(cropped_data)
print(cropped_data.shape)
def make_patches(image,patch_size=16):
    dim,H,W = np.array(image).shape
    num_patches = int((H*W/(patch_size**2)))
    patches = np.zeros((num_patches,patch_size*patch_size,dim)) #(324,256,5)
    patch_idx = 0
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]  # Shape (5, 16, 16)
            patches[patch_idx] = patch.reshape(dim, -1).T  # (256, 5)
            patch_idx += 1
    
    return patches  

class StaticPatchCNN(nn.Module):
    def __init__(self, embed_dim=1280):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 256, embed_dim)  

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(torch.float32)
        x = x.view(324, 5, 256)  # Reshape to (batch=324, channels=5, width=256)
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = x.view(324, -1)  # Flatten
        x = self.fc(x)  # Final transformation to 1280-dim
        return x
    
patches = make_patches(np.array(cropped_data))
model =  StaticPatchCNN(embed_dim=1280)
output = model(patches)
print(output[0])

def final_encoding(tokens,patch_length,dim):
    encoding = torch.zeros(patch_length,dim)
    position = torch.arange(0, patch_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    encoding[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
    encoding[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
    return tokens+encoding

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Convert tensor to numpy for visualization
encoded_numpy = final_encoding(output,324,1280).detach().numpy()# Ensure it's detached from computation graph

smooth_encoding = gaussian_filter(encoded_numpy,sigma=1)

# Plot the smoothed heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(smooth_encoding, cmap="magma", xticklabels=False, yticklabels=False)
plt.title("Smoothed Positional Encoding Heatmap")
#plt.show()

#make the attention block

class Attention_block(nn.Module):
    def __init__ (self,encoding_len = 1280,num_heads = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_len = encoding_len//num_heads

        self.Q_linear = nn.Linear(encoding_len,encoding_len)
        self.K_linear = nn.Linear(encoding_len,encoding_len)
        self.V_linear = nn.Linear(encoding_len,encoding_len)

        self.out_linear = nn.Linear(encoding_len,encoding_len)
    
    def forward(self, x):  # x shape: (324, 1280)
        seq_len, embed_dim = x.shape  

    # Step 1: Linear projections for Q, K, V
        Q = self.Q_linear(x)  # (324, 1280)
        K = self.K_linear(x)  # (324, 1280)
        V = self.V_linear(x)  # (324, 1280)

    # Step 2: Reshape for multi-head attention
        Q = Q.view(seq_len, self.num_heads, self.head_len).transpose(0, 1)  # (8, 324, 160)
        K = K.view(seq_len, self.num_heads, self.head_len).transpose(0, 1)  # (8, 324, 160)
        V = V.view(seq_len, self.num_heads, self.head_len).transpose(0, 1)  # (8, 324, 160)

        # Step 3: Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_len, dtype=torch.float32))  # (8, 324, 324)
        attn_weights = F.softmax(scores, dim=-1)  # Attention map
        attn_output = torch.matmul(attn_weights, V)  # Weighted sum (8, 324, 160)

    # Step 4: Concatenate heads back
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, embed_dim)  # (324, 1280)

    # Step 5: Final linear layer
        output = self.out_linear(attn_output)

        return output  # (324, 1280)

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * mlp_ratio)  # Expand dimension
        self.activation = nn.GELU()  # Non-linearity
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim * mlp_ratio, embed_dim)  # Project back
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,encoding_len=1280,mlp_ratio=4,dropout=0.1,num_heads=8):
        super().__init__()
        self.attention = Attention_block(encoding_len,num_heads)
        self.norm1 = nn.LayerNorm(encoding_len)
        self.mlp = MLPBlock(encoding_len,mlp_ratio,dropout)
        self.norm2 = nn.LayerNorm(encoding_len)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.attention(x)
        x = x + self.dropout(x)
        x = self.norm1(x)
        x = self.mlp(x)
        x = x + self.dropout(x)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, encoding_len=1280, mlp_ratio=4, dropout=0.1, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(encoding_len, mlp_ratio, dropout, num_heads) for _ in range(num_layers)]
        )
        # Adding a dummy vector a=hoping it will store all the important info
        self.dummy_token = nn.Parameter(torch.randn(1,1,encoding_len))

    def forward(self, x):  # Make sure it has (1, 1280)
        x = torch.cat([x,self.dummy_token], dim=0)
        for layer in self.layers:
            x = layer(x) 
        output = x[-1]
        return output
    
class SpectrumDecoder(nn.Module):
    def __init__(self, input_dim=1280, output_dim=3836):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)  # Output matches spectrum length (3836,)
        )
    def forward(self, x):
        return self.decoder(x)  # Output: (3836,)

class Model(nn.Module):
    def __init__(self,transformer_encoder,spec_decoder):
        super().__init__()
        self.encoder  = transformer_encoder
        self.decoder = spec_decoder

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Inititialize the entire model

Transformer_encoder = TransformerEncoder(num_layers=6, encoding_len=1280, mlp_ratio=4, dropout=0.1, num_heads=8)
spec_decoder = SpectrumDecoder( input_dim=1280, output_dim=3836)

model = Model(Transformer_encoder,spec_decoder)

# Check for gpu's
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
import torch.optim as optim
loss_func =  nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 1e-4,weight_decay=1e-5)


encodings = torch.tensor(final_encoding(output,324,1280), dtype=torch.float32).unsqueeze(0).to(device)
#Getting the spectrum

from astropy.table import Table
file_name = "spec-0266-51602-0006.fits"
with fits.open(file_name) as hdul:
    hdul.info()  # Display FITS structure
    coadd_data = Table(hdul[1].data)

wavelength = 10**coadd_data['loglam']  # Convert log wavelength to normal scale
flux = coadd_data['flux'] 
flux = np.asarray(flux)  # Ensure it's a NumPy array
flux = flux.astype(np.float32)  # Convert to float32
flux = np.ascontiguousarray(flux)  # Convert to native byte order

# Now create PyTorch tensor
true_spectrum = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).to(device)
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # No gradient tracking
    predicted_spectrum = model(encodings) 

predicted_spectrum = predicted_spectrum.cpu().numpy().flatten()
true_spectrum = true_spectrum.cpu().numpy().flatten()

plt.figure(figsize=(10, 5))
plt.plot(true_spectrum, label="True Spectrum", color='blue', alpha=0.7)
plt.plot(predicted_spectrum, label="Predicted Spectrum", color='red', linestyle='dashed', alpha=0.7)
plt.legend()
plt.xlabel("Wavelength Index")
plt.ylabel("Flux")
plt.title("Spectrum Reconstruction Test")
plt.show()
