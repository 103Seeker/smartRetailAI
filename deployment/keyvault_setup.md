# Azure Key Vault — Security Setup

## Why Key Vault
All API keys and secrets are stored in Azure Key Vault.
The application never reads secrets from hardcoded values or committed files.
At runtime, the Azure Web App fetches secrets from Key Vault via environment variable references.

## Secrets Stored in Key Vault

| Secret Name              | Description                        |
|--------------------------|------------------------------------|
| AZURE-OPENAI-API-KEY     | Azure OpenAI service key           |
| AZURE-OPENAI-ENDPOINT    | Azure OpenAI resource endpoint     |
| DATABASE-URL             | Production database connection URL |

## Setup Steps

### 1. Create Key Vault
```bash
az keyvault create \
  --name smart-retail-kv \
  --resource-group smart-retail-rg \
  --location eastus
```

### 2. Add Secrets
```bash
az keyvault secret set --vault-name smart-retail-kv --name AZURE-OPENAI-API-KEY --value "<your-key>"
az keyvault secret set --vault-name smart-retail-kv --name AZURE-OPENAI-ENDPOINT --value "<your-endpoint>"
az keyvault secret set --vault-name smart-retail-kv --name DATABASE-URL --value "<your-db-url>"
```

### 3. Link to Azure Web App
```bash
az webapp config appsettings set \
  --name smart-retail-app \
  --resource-group smart-retail-rg \
  --settings AZURE_OPENAI_API_KEY="@Microsoft.KeyVault(SecretUri=https://smart-retail-kv.vault.azure.net/secrets/AZURE-OPENAI-API-KEY/)"
```

### 4. Grant Web App Access to Key Vault
```bash
az keyvault set-policy \
  --name smart-retail-kv \
  --object-id <webapp-managed-identity-id> \
  --secret-permissions get list
```

## Local Development
For local development, copy `.env.example` to `.env` and fill in values.
`.env` is listed in `.gitignore` and is never committed.
