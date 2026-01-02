# Pytorch C++ & RNN

# Installation

Download the zip from pytorch https://pytorch.org/get-started/locally/  
Use the C++/Java and the CPU version since are task are light so no gpu needed for now
Unzip the downloaded zip file into a directory of your choice

# Link folders

ln -s ../Results/ .
ln -s ../Data .
ln -s ../include .


# Train to google cloud

# this is ont to just update the cpp files on VM instance
gcloud compute scp RNN/*.cpp instance-20260102-140421:~/project/RNN/ --zone "europe-west1-b"

# Should not change that much after initial copy
gcloud compute scp RNN/Makefile instance-20260102-140421:~/project/RNN/ --zone "europe-west1-b"
gcloud compute scp RNN/Dockerfile instance-20260102-140421:~/project/RNN/ --zone "europe-west1-b"
gcloud compute scp --recurse include/ instance-20260102-140421:~/project/ --zone "europe-west1-b"
gcloud compute scp --recurse Data/ instance-20260102-140421:~/project/ --zone "europe-west1-b"

# Docker

## FROM THE ROOT DIRECTORY
docker build -f RNN/Dockerfile -t lstm-app .
docker run --rm lstm-app