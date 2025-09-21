# BigDataLab3 - Secret Management with Ansible Vault
This project extends BigDataLab2 by adding secret management using Ansible Vault. Instead of storing database credentials in environment variables or configuration files, they are now securely stored in Vault.

## API Endpoints
* /predict: Make a prediction based on a review
* /train: Make a train of choosen ML model

## CD
CD starts every monday at around 9am

## Security Notes
The .env and config.ini files are only used for initial setup and should be removed in production.
