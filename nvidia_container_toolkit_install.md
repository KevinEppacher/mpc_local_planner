# Nvidia Container Toolkit Installation

For detailed instructions, refer to the official documentation:  
[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**Note:** Make sure the local NVIDIA drivers are installed and working on your system. You should be able to run `nvidia-smi` in a terminal and see information about your GPU before proceeding.

**Step 1: Add the NVIDIA Container Toolkit repository**

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

**Step 2: Update package lists and install the toolkit**

```sh
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

**Step 3: Restart your computer to apply changes.**

**Step 4: Verify the installation**

Open a terminal and run:
```sh
nvidia-smi
```
This command should display information about your