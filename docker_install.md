# Docker Installation Documentation:

You can follow the official Docker installation documentation for Ubuntu by visiting the following link:
[Docker Install Guide for Ubuntu](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

### Step-by-Step Installation:

1. Update your package list:

   ```bash
   sudo apt-get update
   ```

2. Install the required packages:

   ```bash
   sudo apt-get install ca-certificates curl
   ```

3. Add Docker's official GPG key:

   ```bash
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc
   ```

4. Add the Docker repository to your Apt sources:

   ```bash
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

5. Update your package list again:

   ```bash
   sudo apt-get update
   ```

6. Install Docker:

   ```bash
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

7. Verify Docker installation by running the hello-world container:

   ```bash
   sudo docker run hello-world
   ```

## 2. Docker Post-Installation Steps:

After installing Docker, you can follow these post-installation steps:
[Docker Post-Install Guide](https://docs.docker.com/engine/install/linux-postinstall/)

1. Create the Docker group:

   ```bash
   sudo groupadd docker
   ```

2. Add your user to the Docker group:

   ```bash
   sudo usermod -aG docker $USER
   ```

3. Log out and log back in for the changes to take effect.

4. Start a new shell session:

   ```bash
   newgrp docker
   ```

5. Verify Docker is working without `sudo`:

   ```bash
   docker run hello-world
   ```

6. Enable Docker services to start on boot:

   ```bash
   sudo systemctl enable docker.service
   sudo systemctl enable containerd.service
   ```
