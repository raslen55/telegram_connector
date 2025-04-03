import os
from datetime import datetime
from typing import Dict, List, Optional

from stix2 import TLP_WHITE, Identity, Indicator, Relationship
from telethon import TelegramClient
from telethon.tl.types import Message

from pycti import OpenCTIConnectorHelper, StixCoreRelationship


class TelegramCollector:
    def __init__(
        self,
        helper: OpenCTIConnectorHelper,
        api_id: str,
        api_hash: str,
        session_string: str,
    ):
        self.helper = helper
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_string = session_string
        self.client = None

        # Initialize identity for the connector
        self.identity = Identity(
            name="Telegram Threat Intelligence",
            identity_class="organization",
            description="Telegram channels threat intelligence collector",
        )

    async def process_message(self, message: Message, channel_name: str) -> List[Dict]:
        """Process a Telegram message and convert it to STIX objects."""
        stix_objects = []

        # Create an indicator from the message
        indicator = Indicator(
            name=f"Telegram Threat Intel - {channel_name}",
            description=message.message,
            pattern_type="stix",
            pattern="[file:hashes.md5 = 'd41d8cd98f00b204e9800998ecf8427e']",  # Placeholder pattern
            valid_from=datetime.utcfromtimestamp(message.date.timestamp()),
            labels=["telegram", channel_name],
            confidence=self.helper.connect_confidence_level,
            object_marking_refs=[TLP_WHITE],
            created_by_ref=self.identity.id,
            custom_properties={
                "x_opencti_main_observable_type": "Message",
                "x_opencti_detection": True,
                "x_opencti_score": 75,
                "x_telegram_channel": channel_name,
                "x_telegram_message_id": message.id,
            },
        )
        stix_objects.append(indicator.serialize())

        # Create relationships if needed
        # Add more STIX object creation based on message content

        return stix_objects

    async def collect_channel_data(self, channel: str, last_run: Optional[int] = None):
        """Collect data from a Telegram channel."""
        try:
            async for message in self.client.iter_messages(channel):
                # Skip messages older than last run
                if last_run and message.date.timestamp() <= last_run:
                    break

                # Process message and create STIX objects
                stix_objects = await self.process_message(message, channel)

                # Send bundle to OpenCTI
                if stix_objects:
                    bundle = {"type": "bundle", "objects": stix_objects}
                    self.helper.send_stix2_bundle(bundle)

        except Exception as e:
            self.helper.log_error(f"Error collecting data from channel {channel}: {str(e)}")

    async def initialize_client(self):
        """Initialize Telegram client."""
        if not self.client:
            self.client = TelegramClient(
                StringSession(self.session_string),
                self.api_id,
                self.api_hash,
            )
            await self.client.start()

    def run(self, channels: List[str], last_run: Optional[int] = None):
        """Run the collector for all channels."""
        try:
            # Initialize Telegram client
            await self.initialize_client()

            # Process each channel
            for channel in channels:
                self.helper.log_info(f"Processing channel: {channel}")
                await self.collect_channel_data(channel, last_run)

        except Exception as e:
            self.helper.log_error(f"Error in collector run: {str(e)}")
        finally:
            if self.client:
                await self.client.disconnect()
