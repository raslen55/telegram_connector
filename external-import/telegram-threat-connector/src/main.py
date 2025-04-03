import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from lib.telegram_collector import TelegramCollector
from pycti import OpenCTIConnectorHelper, get_config_variable


class TelegramConnector:
    def __init__(self):
        # Initialization of OpenCTI connector helper
        config = {
            "name": "Telegram Threat Intelligence",
            "confidence_level": get_config_variable(
                "CONNECTOR_CONFIDENCE_LEVEL", ["connector", "confidence_level"], 75
            ),
            "update_existing_data": get_config_variable(
                "CONNECTOR_UPDATE_EXISTING_DATA",
                ["connector", "update_existing_data"],
                True,
            ),
            "log_level": get_config_variable(
                "CONNECTOR_LOG_LEVEL", ["connector", "log_level"], "info"
            ),
        }
        self.helper = OpenCTIConnectorHelper(config)

        # Get configuration
        self.telegram_channels = get_config_variable(
            "CONNECTOR_TELEGRAM_CHANNELS", ["connector", "telegram_channels"], ""
        ).split(",")
        self.telegram_api_id = get_config_variable(
            "TELEGRAM_API_ID", ["telegram", "api_id"], None
        )
        self.telegram_api_hash = get_config_variable(
            "TELEGRAM_API_HASH", ["telegram", "api_hash"], None
        )
        self.telegram_session_string = get_config_variable(
            "TELEGRAM_SESSION_STRING", ["telegram", "session_string"], None
        )

        self.collector = TelegramCollector(
            self.helper,
            self.telegram_api_id,
            self.telegram_api_hash,
            self.telegram_session_string,
        )

    def run(self):
        self.helper.log_info("Starting Telegram Threat Intelligence connector...")
        while True:
            try:
                # Get the current timestamp and check
                timestamp = int(time.time())
                current_state = self.helper.get_state()
                if current_state is not None and "last_run" in current_state:
                    last_run = current_state["last_run"]
                    self.helper.log_info(
                        f"Connector last run: {datetime.fromtimestamp(last_run)}"
                    )
                else:
                    last_run = None
                    self.helper.log_info("Connector has never run")

                # Run the collector
                self.collector.run(self.telegram_channels, last_run)

                # Store the current timestamp as a last run
                self.helper.set_state({"last_run": timestamp})

                # Sleep for 1 hour
                time.sleep(3600)

            except (KeyboardInterrupt, SystemExit):
                self.helper.log_info("Connector stop")
                sys.exit(0)
            except Exception as e:
                self.helper.log_error(str(e))
                time.sleep(60)


if __name__ == "__main__":
    try:
        connector = TelegramConnector()
        connector.run()
    except Exception as e:
        print(e)
        time.sleep(10)
        sys.exit(0)
