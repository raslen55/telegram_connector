# OpenCTI Telegram Threat Intelligence Connector

This connector allows you to import threat intelligence data from Telegram channels into OpenCTI.

## Installation

The connector can be installed directly from the OpenCTI platform.

### Requirements

- OpenCTI Platform >= 5.12.5
- Python >= 3.11

### Configuration

| Parameter | Docker envvar | Mandatory | Description |
| --- | --- | --- | --- |
| OpenCTI URL | `OPENCTI_URL` | Yes | The URL of the OpenCTI platform |
| OpenCTI Token | `OPENCTI_TOKEN` | Yes | The token of the OpenCTI user |
| Connector ID | `CONNECTOR_ID` | Yes | A unique UUIDv4 for this connector |
| Connector Type | `CONNECTOR_TYPE` | Yes | Must be `EXTERNAL_IMPORT` |
| Connector Name | `CONNECTOR_NAME` | Yes | The name of the connector |
| Connector Scope | `CONNECTOR_SCOPE` | Yes | Must be `telegram` |
| Connector Confidence Level | `CONNECTOR_CONFIDENCE_LEVEL` | Yes | The default confidence level (0-100) |
| Connector Update Existing Data | `CONNECTOR_UPDATE_EXISTING_DATA` | Yes | Whether to update existing data |
| Connector Log Level | `CONNECTOR_LOG_LEVEL` | Yes | The log level (debug, info, warn, error) |
| Telegram Channels | `CONNECTOR_TELEGRAM_CHANNELS` | Yes | Comma-separated list of Telegram channels to monitor |
| Telegram API ID | `TELEGRAM_API_ID` | Yes | Your Telegram API ID |
| Telegram API Hash | `TELEGRAM_API_HASH` | Yes | Your Telegram API Hash |
| Telegram Session String | `TELEGRAM_SESSION_STRING` | Yes | Your Telegram session string |

### Docker Deployment

1. Create a `.env` file with the required configuration
2. Run `docker-compose up -d`

## Usage

The connector will:
1. Connect to specified Telegram channels
2. Collect messages and convert them to STIX format
3. Import the data into OpenCTI
4. Run every hour to collect new messages

## Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your configuration
4. Run the connector: `python src/main.py`
