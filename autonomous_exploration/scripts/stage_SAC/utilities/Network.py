import torch

def conv2d_size_out(size, kernel_size = 5, stride = 2):
    return (size - (kernel_size - 1) - 1) // stride  + 1


class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Network, self).__init__()

        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1,32,kernel_size=5,stride=2),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32,16,kernel_size=5,stride=2),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16,8,kernel_size=5,stride=2),
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU()
                )
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_dimension[1][1])))  # image 96x96
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_dimension[1][2])))

        linear_input_size = (convw * convh * 8) + input_dimension[0][0]*input_dimension[0][1]
                    # CNN flatten size (convw*conh*LastOutput) + Frontier flatten size


        self.layer_1 = torch.nn.Linear(in_features=linear_input_size, out_features=512)
        self.layer_2 = torch.nn.Linear(in_features=512, out_features=512)
        self.output_layer = torch.nn.Linear(in_features=512, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        x1 = inpt[0].view(inpt[0].size(0), -1) #flatten frontier relative polar coord

        x2 = self.conv(inpt[1]) # CNN map image
        x2 = x2.view(x2.size(0), -1) #flatten CNN

        x = torch.cat((x1, x2), dim=1)  # concat inputs (1,656)


        layer_1_output = torch.nn.functional.relu(self.layer_1(x))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_activation(self.output_layer(layer_2_output))
        return output

