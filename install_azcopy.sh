# Download AzCopy binary directly
cd /tmp
wget https://aka.ms/downloadazcopy-v10-linux

# Extract it
tar -xvf downloadazcopy-v10-linux

# Move to system path
sudo cp azcopy_linux_amd64_*/azcopy /usr/local/bin/

# Make it executable
sudo chmod +x /usr/local/bin/azcopy

# Verify installation
azcopy --version

# Clean up
rm -rf downloadazcopy-v10-linux azcopy_linux_amd64_*