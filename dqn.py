import torch
import torch.nn as nn

class DQN(nn.Module) :
    def __init__(self , input_shape , num_actions):
        super(DQN,self).__init__()

        c , h , w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c , 32 , kernel_size = 8,stride = 4 ),
            nn.ReLU(),

            nn.Conv2d(32,64,kernel_size = 4 , stride = 2),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size = 3 , stride = 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136 , 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self , x):
        x = self.features(x)
        x = self.fc(x)
        return x