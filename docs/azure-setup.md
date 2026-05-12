# Azure ML Workspace Setup

Follow these steps once to connect your local machine to the team Azure ML workspace.

## Prerequisites

- Python 3.10+ with your project virtualenv active (or miniconda)
- Git with the repo cloned

---

## 1. Install Azure CLI

Open **PowerShell as Administrator** and run:

```powershell
winget install Microsoft.AzureCLI
```

Close and reopen your terminal after install so `az` is on your PATH. Verify:

```powershell
az --version
```

---

## 2. Log in to Azure

```powershell
az login
```

A browser window will open. Sign in with your **BUas account** (`@buas.nl`).

When prompted to select a subscription/tenant, press **Enter** to accept the default, which should be:

```
Azure subscription 1  |  0a94de80-6d3b-49f2-b3e9-ec5818862801  |  BUas
```

Confirm your login:

```powershell
az account show
```

You should see `"name": "242817@buas.nl"` (or your own BUas email) and the BUas tenant.

---

## 3. Install Python dependencies

```powershell
pip install azure-ai-ml azure-identity
```

---

## 4. Create your config.json

Copy the template and fill in your real values (they are the same for everyone on the team):

```powershell
cp config.example.json config.json
```

Edit `config.json`:

```json
{
  "subscription_id": "0a94de80-6d3b-49f2-b3e9-ec5818862801",
  "resource_group": "buas-y2",
  "workspace_name": "team-CV5-2026"
}
```

> `config.json` is in `.gitignore` — never commit it.

---

## 5. Test the connection

```powershell
python scripts/test_azure_connection.py
```

Expected output:

```
Connected to workspace: team-CV5-2026
  Location      : westeurope
  Resource group: buas-y2
```

If it errors, the most common cause is that `az` is not on your PATH in the current terminal. Open a **new** terminal and try again.

---

## Workspace details

| Field | Value |
|---|---|
| Subscription ID | `0a94de80-6d3b-49f2-b3e9-ec5818862801` |
| Resource group | `buas-y2` |
| Workspace name | `team-CV5-2026` |
| Location | West Europe |
| Portal | [ml.azure.com](https://ml.azure.com) |
